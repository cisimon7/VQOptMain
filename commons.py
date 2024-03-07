import os
import shutil
import random
import numpy as np
from torchinfo import summary
from omegaconf import DictConfig

import torch as th
from model.VQVAE import VQVAEModule, VQVAECNN, VQVAELinear
from model.PixelLinear import PixelLinear
from model.PixelCNN import PixelCNN, PixObsNet, PixObsNetCNN


def get_vqvae(vqvae_cfg: DictConfig, mean, std, device="cpu") -> VQVAEModule:

    seed_mch()

    vqvae = None
    if vqvae_cfg.type == "cnn":
        vqvae = VQVAECNN(
            mean=mean, std=std, 
            out_dim=vqvae_cfg.encoder.output_dim,
            hid_channels=vqvae_cfg.encoder.hid_channels,
            out_channels=vqvae_cfg.codebook.num_features,
            kernel_size=vqvae_cfg.encoder.kernel_size,
            padding=vqvae_cfg.encoder.padding,
            num_layers=vqvae_cfg.encoder.num_layers,
            num_embeddings=vqvae_cfg.codebook.num_embeddings,
            embedding_dim=vqvae_cfg.codebook.latent_dim,
            decoder_shapes=vqvae_cfg.decoder.shape,
            init_in_dim=vqvae_cfg.initializer.input_dim,
            init_out_dim=vqvae_cfg.initializer.output_dim,
            init_shape=vqvae_cfg.initializer.shape,
            device=device, num_batch=vqvae_cfg.batch_size
        )
    elif vqvae_cfg.type == "linear":
        vqvae = VQVAELinear(
            inp_dim=vqvae_cfg.input_dim,
            out_dim=vqvae_cfg.output_dim,
            num_embeddings=vqvae_cfg.num_embeddings,
            embedding_dim=vqvae_cfg.latent_dim,
            encoder_shapes=vqvae_cfg.encoder_shape,
            num_features=vqvae_cfg.num_features,
            decoder_shapes=vqvae_cfg.decoder_shape,
            device=device, num_batch=vqvae_cfg.batch_size
        )

    for key, value in vqvae_cfg.optimizer.items():
        assert hasattr(vqvae.dOptimizer, key)
        setattr(vqvae.dOptimizer, key, value)

    print("*****************************************************************************")
    summary(vqvae, [(vqvae_cfg.batch_size, 200), (vqvae_cfg.batch_size, 55)], device=device, mode="train")
    return vqvae


def get_pixcnn(vqvae_cfg: DictConfig, pix_cfg: DictConfig, device="cpu"):
    pix_net = PixelCNN(
        num_embedding=vqvae_cfg.codebook.num_embeddings,
        kernel_size=pix_cfg.conv_kernel_size, in_channels=1, padding=pix_cfg.conv_padding,
        n_channels=pix_cfg.n_channels, n_layers=pix_cfg.n_layers, device=device
    )
    return pix_net


def get_pixlin(vqvae_cfg: DictConfig, pix_cfg: DictConfig, device="cpu"):
    pix_net = PixelLinear(
        num_embedding=vqvae_cfg.num_embeddings,
        in_feature=vqvae_cfg.num_features, in_channels=1,
        out_channels=pix_cfg.n_channels, n_layers=pix_cfg.n_layers, device=device
    )
    return pix_net


def get_pix(vqvae_cfg: DictConfig, pix_cfg: DictConfig, mean, std, device="cpu"):
    obs_net = None
    if pix_cfg.type == "cnn":
        obs_net = PixObsNetCNN(
            mean=mean, std=std, hid_channels=pix_cfg.obs_cnn.hid_channels,
            kernel_size=pix_cfg.obs_cnn.kernel_size, padding=pix_cfg.obs_cnn.padding,
            num_layers=pix_cfg.obs_cnn.num_layers, output_shape=vqvae_cfg.codebook.num_features, device=device,
        )
        print("*****************************************************************************")
        summary(obs_net, [(1, 55)], device=device)
    elif pix_cfg.type == "linear":
        obs_net = PixObsNet(
            mean=mean, std=std, layers=pix_cfg.obs_layers, 
            input_shape=pix_cfg.cond_dim, output_shape=vqvae_cfg.codebook.num_features,
            device=device,
        )
        print("*****************************************************************************")
        summary(obs_net, [(1, 55)], device="cuda")

    if pix_cfg.type == "cnn":
        pixel_cnn = get_pixcnn(vqvae_cfg, pix_cfg, device)
        print("*****************************************************************************")
        summary(pixel_cnn, [(1, 1, 7), (1, 1, 7)], device=device)
        return pixel_cnn, obs_net
    elif pix_cfg.type == "linear":
        return get_pixlin(vqvae_cfg, pix_cfg, device), obs_net
    
    
def make_dir(path: str, clean=True):
    if os.path.exists(path) and clean:
        shutil.rmtree(path)
        
    os.makedirs(path)


def seed_mch():
    th.set_default_dtype(th.float32)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.set_default_dtype(th.float32)
    th.set_float32_matmul_precision('high')
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    