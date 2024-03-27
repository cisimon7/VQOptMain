import hydra
from omegaconf import DictConfig

import sys
import gym
import highway_env
from gym.wrappers import RecordVideo

from pprint import pprint
from time import perf_counter

from tqdm import tqdm
import numpy as np
import torch as th
from utils.visualizations import PlotEnv
from datasets.datasets2 import get_dataset
from commons import get_vqvae, get_pix, seed_mch
from utils.config import config, print_config_details, print_eps_details

import warnings
warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")

import logging
logging.getLogger("jax").setLevel(logging.WARNING)


class InitializerModule2(th.nn.Module):
    def __init__(self, mean, std, in_dim, out_dim, mlp_shape):
        super(InitializerModule2, self).__init__()
        self.mean = mean
        self.std = std
        shapes = [in_dim] + mlp_shape + [out_dim+1]
        self.net = th.nn.Sequential(
        *[
            th.nn.Sequential(
                th.nn.Linear(in_fea, out_fea),
                th.nn.BatchNorm1d(out_fea),
                th.nn.ReLU()
            )
            for i, (in_fea, out_fea) in enumerate(th.tensor(shapes).unfold(0, 2, 1))
        ] + [
            th.nn.Linear(shapes[-1], shapes[-1])
        ])
            
    def forward(self, primal_sol_1, observations):
        observations = (observations - self.mean) / self.std
        x = th.cat([primal_sol_1, observations], dim=-1)           
        out = self.net(x)
        out[..., -1] = th.nn.functional.sigmoid(out[..., -1])
        return out


@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig) -> None:
    seed_mch()
    
    cfg.pixel.batch_size = cfg.batch_size
    cfg.vqvae.optimizer = cfg.optimizer
    cfg.vqvae.batch_size = cfg.batch_size
    mean, std = 49.2430, 81.9682
    
    vqvae = get_vqvae(cfg.vqvae, mean, std, cfg.device)
    vqvae.dOptimizer.initializer_model = InitializerModule2(
        mean, std, 
        cfg.vqvae.initializer.input_dim,
        cfg.vqvae.initializer.output_dim,
        cfg.vqvae.initializer.shape
    )
    vqvae.dOptimizer.initializer_model.to(cfg.device)
    vqvae.load_state_dict(th.load(f"./weights/main/{cfg.vqvae.type}/{cfg.main_name}"))
    
    pix_net, obs_net = get_pix(cfg.vqvae, cfg.pixel, mean, std, cfg.device)
    pix_net.load_state_dict(th.load(f"./weights/pixel/{cfg.pixel.type}/{cfg.pix_name}"))
    obs_net.load_state_dict(th.load(f"./weights/pixel/{cfg.pixel.type}/{cfg.obs_name}"))
    
    vqvae.eval(), pix_net.eval(), obs_net.eval()
    compiled_pix_net, compiled_obs_net = th.compile(pix_net), th.compile(obs_net)
    vqvae.init_jax_functions(cfg.vqvae)

    @th.compile
    def sample(observations):
        latent_sample = th.zeros(
            cfg.batch_size, 1, cfg.vqvae.codebook.num_features, dtype=th.float32, device=cfg.device
        )
        cond = compiled_obs_net(observations)
        for k in range(cfg.vqvae.codebook.num_features):
            output = compiled_pix_net(latent_sample, cond.repeat(cfg.batch_size, 1, 1))
            latent_sample[..., k] = th.multinomial(
                th.nn.functional.softmax(output[..., k], dim=-1),
                num_samples=1, replacement=False
            )
        
        return latent_sample.squeeze(dim=-2)

    config["vehicles_density"] = cfg.env.density
    config["speed_limit"] = cfg.env.speed_limit
    env = gym.make("highway-v0", config=config)
    env.seed(cfg.env.seed)
    render = cfg.env.render
    plot = cfg.env.plot
    rec = cfg.env.record
    
    if rec:
        VIDEO_PATH = f"./videos/{cfg.main_name[:-4]}"
        env = RecordVideo(
            env, video_folder=f"{VIDEO_PATH}/{env.config['vehicles_density']}_density_{cfg.env.seed}_seed _{cfg.optimizer.v_des}_vdes_{cfg.env.name}",
            episode_trigger=lambda e: True
        )
        env.unwrapped.set_record_video_wrapper(env)
    
    crash_count = 0
    all_vels = []

    plot_env = None
    if plot:
        obs = np.zeros(55)
        plot_env = PlotEnv(
            np.stack([obs[5::5], obs[6::5], obs[9::5]], axis=-1), 
            batch_size=cfg.batch_size
        )
    

    with th.no_grad():
        for i in range(cfg.env.num_eps):
            obs, done = env.reset(), False
            vels = []
            
            start_time = perf_counter()
            while not done:
                speed = np.linalg.norm(obs[2:4])
                vels.append(speed)
                
                observation = th.from_numpy(obs).unsqueeze(dim=0).to(device=cfg.device, dtype=th.float32)
                latent = sample(observation)
                latent = vqvae.quantizer.embedding(latent.to(dtype=th.int32))
                control, all_trajs, opt_traj = vqvae.decode_jax(
                    latent, observation.repeat(cfg.batch_size, 1).cpu().numpy()
                )
                obs, reward, done, info = env.step(control)
                
                if render:
                    env.render()

                if plot:
                    plot_env.step(
                        np.hstack([obs[0], obs[1], obs[4]]),
                        np.stack([obs[5::5], obs[6::5], obs[9::5]], axis=-1),
                        np.asarray(obs[:2]), np.asarray(all_trajs), None, np.asarray(opt_traj)
                    )
                    
            duration = perf_counter() - start_time
                
            crashed = info["crashed"]
            if crashed:
                crash_count += 1
            else:
                all_vels.append(vels)
            
            print_eps_details(
                step=i, crashed=crashed, mean_speed=np.mean(vels),
                duration=duration, collision_rate=100*crash_count/(i+1)
            )

    np.savez(f"speed_{config['vehicles_density']}_{config['speed_limit']}_{cfg.optimizer.v_des}_{cfg.optimizer.maxiter}", np.asarray(all_vels))    
    print_config_details(
        config, (100*crash_count/(i+1)), 
        np.mean(all_vels).round(2), np.std(all_vels).round(2),
        seed=cfg.env.seed, space_width=10
    )
                    

if __name__ == "__main__":
    main()
