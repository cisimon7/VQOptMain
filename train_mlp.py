import hydra
from tqdm import tqdm
from omegaconf import DictConfig

import sys

import random
import numpy as np
import torch as th
from datasets.datasets2 import get_dataset
from utils.visualizations import view_train
from torch.utils.tensorboard import SummaryWriter
from commons import get_vqvae, make_dir, seed_mch
from torch.utils.data import TensorDataset, DataLoader, Dataset


class TriDataset(Dataset):
    def __init__(self, obs, traj, primal):
        self.inp, self.out, self.primal = obs, primal, traj

    def __len__(self):
        return len(self.inp)

    def __getitem__(self, idx):
        inp = self.inp[idx]
        out = self.out[idx]
        return (inp.float(), self.primal[idx]), out.float()


@hydra.main(version_base=None, config_path="./configs", config_name="mlp")
def main(cfg: DictConfig) -> None:
    seed_mch()
    
    primals_1 = th.load("./datasets/dist_prims1_04_00.pt")
    observations = th.load("./datasets/dist_obs_04_00.pt").repeat_interleave(primals_1.size(1), dim=0)
    targets = th.load("./datasets/dist_targets_04_00.pt").repeat_interleave(primals_1.size(1), dim=0)
    primals_1 = primals_1.flatten(start_dim=0, end_dim=1)
    
    train_dataset = TriDataset(observations, targets, primals_1)
    train_loader = DataLoader(train_dataset, batch_size=min(cfg.batch_size, len(train_dataset)), shuffle=True, num_workers=0, drop_last=True, worker_init_fn=lambda id_: seed_mch())
    _, _, mean, std = get_dataset(cfg.vqvae.batch_size, test_ratio=0.0)
    print("Dataset Loaded")

    type_name = cfg.vqvae.type
    cfg.vqvae.batch_size = cfg.batch_size
    vqvae = get_vqvae(cfg.vqvae, mean, std, cfg.device)
    vqvae.load_state_dict(th.load(f"./weights/vqvae/{cfg.vqvae.type}/{cfg.vqvae_name}"))
    
    for param in vqvae.encoder.parameters():
        param.requires_grad = False
        
    for param in vqvae.decoder.parameters():
        param.requires_grad = False

    for param in vqvae.quantizer.parameters():
        param.requires_grad = False
        
    compiled_vqvae = th.compile(vqvae)
    
    optimizer = th.optim.AdamW(vqvae.dOptimizer.parameters(), lr=cfg.learning_rate, weight_decay=6e-5)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, verbose=True)
    
    writer = SummaryWriter(f"{cfg.train_log_dir}/pixel")
    model_save_path = f"./weights/main/{type_name}/{cfg.train_name}"
    train_imgs_path = f"./train_imgs/mlp/{type_name}/{cfg.train_name}"
    make_dir(model_save_path, clean=False)
    make_dir(train_imgs_path)

    print("train_name: ", cfg.train_name)
    print("maxiter: ", vqvae.dOptimizer.maxiter)
    print("num_batch: ", vqvae.dOptimizer.num_batch)
    print("Starting Training MLP")
    
    obs, targets, qp2_output = None, None, None
    seed_mch()
    for epoch in range(1, cfg.vqvae.num_epoch + 1):
        vqvae.train()
        losses_list = ([],)
        for (obs, targets), primals_1 in tqdm(train_loader):
            
            obs, primals_1 = (obs.to(device=cfg.device), primals_1.to(device=cfg.device))
            qp2_output = compiled_vqvae.dOptimizer.forward_q2(obs, primals_1)

            loss = compiled_vqvae.compute_mlp_loss(obs, qp2_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for (i, loss) in enumerate([loss]):
                losses_list[i].append(loss.detach())
            
            traj2 = qp2_output[0][0].detach().cpu()

        scheduler.step(th.stack(losses_list[0]).mean().detach())
        # scheduler.print_lr(True, optimizer, scheduler.get_last_lr(), epoch)
        writer.add_scalar(f"{cfg.pixel.train_name}/{cfg.train_name}/loss_aug", th.stack(losses_list[0]).detach().mean(), epoch)

        view_train(
            obs[0].detach().cpu(),
            targets[0].reshape(2, -1).mT.detach().cpu(),
            traj2.reshape(2, -1).mT.detach().cpu(),
            save_path=f"{train_imgs_path}/{epoch:03d}.png"
        )
        
        if epoch % 50 == 0:
            th.save(vqvae.state_dict(), f"{model_save_path}/model_{epoch:02d}.pth")
            
  
if __name__ == "__main__":
    main()
