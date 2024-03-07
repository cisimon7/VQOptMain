import hydra
from omegaconf import DictConfig

import gym
import highway_env
from gym.wrappers import RecordVideo

import numpy as np
import torch as th
from tqdm import tqdm
from utils.visualizations import PlotEnv
from datasets.datasets2 import get_dataset
from commons import get_vqvae, get_pix, seed_mch
from utils.config import config, print_config_details, print_eps_details


@hydra.main(version_base=None, config_path="./configs", config_name="main")
def main(cfg: DictConfig) -> None:
    seed_mch()
    
    cfg.batch_size = 10
    cfg.pixel.batch_size = cfg.batch_size
    cfg.vqvae.optimizer = cfg.optimizer
    cfg.vqvae.batch_size = cfg.batch_size
    train_loader, _, mean, std = get_dataset(500_000, test_ratio=0)
    
    vqvae = get_vqvae(cfg.vqvae, mean, std, cfg.device)
    vqvae.dOptimizer.maxiter = 1  # only interested in primal_sol_1
    vqvae.load_state_dict(th.load(f"./weights/vqvae/{cfg.vqvae.type}/{cfg.vqvae_name}"))
    
    pix_net, obs_net = get_pix(cfg.vqvae, cfg.pixel, mean, std, cfg.device)
    pix_net.load_state_dict(th.load(f"./weights/pixel/{cfg.pixel.type}/{cfg.pix_name}"))
    obs_net.load_state_dict(th.load(f"./weights/pixel/{cfg.pixel.type}/{cfg.obs_name}"))
    
    vqvae.eval(), pix_net.eval(), obs_net.eval()
    compiled_vqvae, compiled_pix_net, compiled_obs_net = (
        th.compile(vqvae), th.compile(pix_net), th.compile(obs_net)
    )
    
    @th.compile
    def sample(observations):
        latent_sample = th.zeros(
            cfg.batch_size, 1, cfg.vqvae.codebook.num_features, dtype=th.float32, device=cfg.device
        )
        cond = obs_net(observations)
        for k in range(cfg.vqvae.codebook.num_features):
            output = pix_net(latent_sample, cond.repeat(cfg.batch_size, 1, 1))
            latent_sample[..., k] = th.multinomial(
                th.nn.functional.softmax(output[..., k], dim=-1),
                num_samples=1, replacement=False
            )
        
        return latent_sample.squeeze(dim=-2)

    env = gym.make("highway-v0", config=config)
    env.seed(cfg.env.seed)
    render = cfg.env.render
    plot = cfg.env.plot
    rec = cfg.env.record
    
    observations = []
    primals_1 = []
    targets = []

    train_dataset = next(iter(train_loader))
    with th.no_grad():
        for obs, target in tqdm(zip(train_dataset[0], train_dataset[1]), total=len(train_dataset[0])):
            observation = obs.unsqueeze(dim=0).to(device=cfg.device, dtype=th.float32)
            latent = sample(observation)
            latent = compiled_vqvae.quantizer.embedding(latent.to(dtype=th.int32))
            trajs, primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch, v_init = compiled_vqvae.decode(
                latent, observation.repeat(cfg.batch_size, 1), inference=False
            )
            
            targets.append(target.detach().cpu())
            observations.append(obs.detach().cpu())
            primals_1.append(primal_sol_level_1.detach().cpu())
            
        observations = th.stack(observations)
        primals_1 = th.stack(primals_1)
        targets = th.stack(targets)
        
        th.save(observations, './datasets/dist_data/dist_obs_05_00.pt') 
        th.save(primals_1, './datasets/dist_data/dist_prims1_05_00.pt')
        th.save(targets, './datasets/dist_data/dist_targets_05_00.pt')
        
        print("size of observations: ", observations.size())
        print("size of primals_1: ", primals_1.size())
        print("size of targets: ", targets.size())


if __name__ == "__main__":
    main()
    