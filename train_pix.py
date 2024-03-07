import hydra
from tqdm import tqdm
from omegaconf import DictConfig

import torch as th
from datasets.datasets2 import get_dataset
from torch.utils.tensorboard import SummaryWriter
from commons import get_vqvae, get_pix, make_dir, seed_mch

# import warnings
# warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")


@hydra.main(version_base=None, config_path="./configs", config_name="train")
def main(cfg: DictConfig) -> None:
    seed_mch()

    train_loader, test_loader, mean, std = get_dataset(cfg.pixel.batch_size, test_ratio=0)
    print("Dataset Loaded")

    cfg.vqvae.batch_size = cfg.pixel.batch_size
    vqvae = get_vqvae(cfg.vqvae, mean, std, cfg.device)
    vqvae.load_state_dict(th.load(f"./weights/vqvae/{cfg.vqvae.type}/{cfg.pixel.vqvae_model}"))
    for param in vqvae.parameters():
        param.requires_grad = False
    
    type_name = cfg.pixel.type
    pix_net, obs_net = get_pix(cfg.vqvae, cfg.pixel, mean, std, cfg.device)

    compiled_vqvae = th.compile(vqvae)
    compiled_pix_net = th.compile(pix_net)
    compiled_obs_net = th.compile(obs_net)

    optimizer = th.optim.AdamW(
        list(pix_net.parameters()) + list(obs_net.parameters()),
        lr=cfg.pixel.learning_rate, weight_decay=6e-5
    )
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=3, verbose=True)
    writer = SummaryWriter(f"{cfg.train_log_dir}/pixel")
    model_save_path = f"./weights/pixel/{type_name}/{cfg.pixel.train_name}"
    make_dir(f"{model_save_path}/", clean=False)
    print("Models Loaded")

    th.autograd.set_detect_anomaly(True)
    for epoch in range(cfg.pixel.num_epoch):
        ep_loss = []
        for observations, targets in tqdm(train_loader):

            observations, targets = (
                observations.to(device=cfg.device),
                targets.to(device=cfg.device)
            )
            _, quantized_indices, _ = compiled_vqvae.encode(targets)  # 5f
            quantized_indices = quantized_indices.unsqueeze(dim=-2)   # 1 x 5f

            cond = compiled_obs_net(observations)  # 1 x 5f
            output = compiled_pix_net(quantized_indices.float(), cond)  # 10m x 5f
            loss = th.nn.functional.cross_entropy(output, quantized_indices.squeeze(dim=-2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_loss.append(loss)

        mean_loss = th.stack(ep_loss).mean().detach()
        scheduler.step(mean_loss)
        writer.add_scalar(f"{cfg.pixel.train_name}/loss_pixel", mean_loss, epoch)

        if epoch % 50 == 0:
            th.save(pix_net.state_dict(), f"{model_save_path}/pixnet_{epoch:02d}.pth")
            th.save(obs_net.state_dict(), f"{model_save_path}/obsnet_{epoch:02d}.pth")


if __name__ == "__main__":
    main()
    