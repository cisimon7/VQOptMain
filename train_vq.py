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


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig) -> None:
    seed_mch()

    cfg.vqvae.optimizer.maxiter = 1
    train_loader, test_loader, mean, std = get_dataset(cfg.vqvae.batch_size, test_ratio=0.0)
    print("Dataset Loaded")

    type_name = cfg.vqvae.type
    vqvae = get_vqvae(cfg.vqvae, mean, std, cfg.device)
    if cfg.vqvae.warm_star:
        vqvae.load_state_dict(th.load(f"./weights/vqvae/{cfg.vqvae.type}/{cfg.vqvae.warm_star}"))
        print(f"warm started using {cfg.vqvae.warm_star}")
        
    vqvae.dOptimizer.qp1_train = True
    for param in vqvae.dOptimizer.initializer_model.parameters():
        param.requires_grad = False
        
    compiled_vqvae = th.compile(vqvae)
    
    optimizer = th.optim.AdamW(vqvae.parameters(), lr=cfg.vqvae.learning_rate, weight_decay=6e-5)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, verbose=True)
    
    writer = SummaryWriter(f"{cfg.train_log_dir}/vqvae")
    model_save_path = f"./weights/vqvae/{type_name}/{cfg.vqvae.train_name}"
    train_imgs_path = f"./train_imgs/vqvae/{type_name}/{cfg.vqvae.train_name}"
    make_dir(model_save_path, clean=False)
    make_dir(train_imgs_path)

    print("train_name: ", cfg.vqvae.train_name)
    print("maxiter: ", vqvae.dOptimizer.maxiter)
    print("num_batch: ", vqvae.dOptimizer.num_batch)
    print("Starting Training")
    
    observations, targets, opt_output = None, None, None
    step, beta = 0, 2
    seed_mch()
    for epoch in range(1, cfg.vqvae.num_epoch + 1):
        vqvae.train()
        losses_list = ([], [], [],)
        i = 0
        for observations, targets in tqdm(train_loader):
            
            observations, targets = (
                observations.to(device=cfg.device),
                targets.to(device=cfg.device)
            )
            opt_output, quantizer_loss = compiled_vqvae(targets, observations)

            loss, rec_loss, quantizer_loss = compiled_vqvae.compute_loss(
                observations, targets, opt_output, quantizer_loss, step, beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for (i, loss) in enumerate([loss, rec_loss, quantizer_loss]):
                losses_list[i].append(loss.detach())
            
            traj2 = opt_output[0][0].detach().cpu()

        if test_loader is not None:
            vqvae.eval()
            with th.no_grad():
                losses_eval = ([], [], [], [])
                for observations, targets in tqdm(test_loader):
                    observations, targets = (
                        observations.to(device=cfg.device),
                        targets.to(device=cfg.device)
                    )
                    opt_output, quantizer_loss = compiled_vqvae(targets, observations)

                    loss, rec_loss, quantizer_loss, aug_loss = compiled_vqvae.compute_loss(
                        observations, targets, opt_output, quantizer_loss, step, beta
                    )

                    for (i, loss) in enumerate([loss, rec_loss, quantizer_loss, aug_loss]):
                        losses_eval[i].append(loss)

                    traj2 = opt_output[0][0].detach().cpu()
                    del loss, rec_loss, aug_loss, quantizer_loss
                    del opt_output

        scheduler.step(th.stack(losses_list[0]).mean().detach())
        for (i, name) in enumerate(["total", "recon", "quantize"]):
            if test_loader is not None:
                writer.add_scalars(
                    f"{cfg.vqvae.train_name}/{name}_loss_{type_name}",
                    {
                        "train": th.stack(losses_list[i]).detach().mean(),
                        "eval": th.stack(losses_eval[i]).detach().mean()
                    },
                    epoch
                )
            else:
                writer.add_scalars(
                    f"{cfg.vqvae.train_name}/{name}_loss_{type_name}",
                    {
                        "train": th.stack(losses_list[i]).detach().mean(),
                    },
                    epoch
                )

        step += 0.5
        view_train(
            observations[0].detach().cpu(),
            targets[0].reshape(2, -1).mT.detach().cpu(),
            traj2.reshape(2, -1).mT.detach().cpu(),
            save_path=f"{train_imgs_path}/{epoch:03d}.png"
        )
        
        if epoch % 50 == 0:
            th.save(vqvae.state_dict(), f"{model_save_path}/model_{epoch:02d}.pth")
            
  
if __name__ == "__main__":
    main()
