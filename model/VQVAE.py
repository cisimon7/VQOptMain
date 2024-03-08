import torch as th
from typing import List
from torch import Tensor
from torchinfo import summary
from model.DOptimizer import DOptimizer, InitializerModule
from model.DOptimizerJax import DOptimizerJax
from utils.torchModel2Jax import InitializerModuleJax
    
    
class VectorQuantizerLayer(th.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, beta=0.2):
        super(VectorQuantizerLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.embedding = th.nn.Embedding(self.num_embeddings, self.embedding_dim, dtype=th.float32)
        # th.nn.init.uniform_(self.embedding.weight, -1/self.num_embeddings, 1/self.num_embeddings)

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

    def get_code_indices(self, x: Tensor):
        """
        x: tensor of shape [..., self.embedding_dim]
        return: tensor of shape [...] 
        """
        x_flat = x.reshape(-1, self.embedding_dim)
        distances = th.cdist(x_flat, self.embedding.weight)
        distances = distances.argmin(dim=-1, keepdims=True)
        return distances.reshape(x.size()[:-1])

    def forward(self, x: th.Tensor):         # [..., self.embedding_dim]
        indices = self.get_code_indices(x)   # [...]
        quantized = self.embedding(indices)  # [..., self.embedding_dim]
        
        commitment_loss = th.nn.functional.mse_loss(quantized.detach(), x)
        codebook_loss = th.nn.functional.mse_loss(quantized, x.detach())

        loss = (self.beta * commitment_loss) + codebook_loss
        quantized = x + (quantized - x).detach()
        
        return quantized, indices, loss
    

class VQVAEModule(th.nn.Module):
    def __init__(
            self, mean, std, out_dim: int,
            num_embeddings: int, embedding_dim: int, num_features: int,
            decoder_shapes: List[int], init_in_dim: int, init_out_dim: int,
            init_shape: List[int], beta=0.2, device="cpu", num_batch=1_000
    ):
        super(VQVAEModule, self).__init__()
        self.num_batch = num_batch
        self.mean, self.std, self.init_shape = mean, std, init_shape
        decoder_shapes = [num_features*embedding_dim] + decoder_shapes + [out_dim]
        
        self.quantizer = VectorQuantizerLayer(num_embeddings, embedding_dim, beta)
        self.decoder = th.nn.Sequential(*[
            th.nn.Sequential(
                th.nn.Linear(in_fea, out_fea),
                th.nn.BatchNorm1d(out_fea),
                th.nn.ReLU()
            )
            for i, (in_fea, out_fea) in enumerate(th.tensor(decoder_shapes).unfold(0, 2, 1))
        ] + [th.nn.Linear(decoder_shapes[-1], decoder_shapes[-1])])
        
        self.dOptimizer = DOptimizer(
            initializer_model=InitializerModule(
                mean, std, init_in_dim, init_out_dim, init_shape
            ),
            num_batch=num_batch, device=device
        )
        
        self.device = device
        self.to(device=device)
    
    def encode(self, target):
        enc_out = self.encoder(target)
        enc_out = enc_out.reshape(*enc_out.size()[:-1], -1, self.quantizer.embedding_dim)
        quantized_out, indices, quantizer_loss = self.quantizer(enc_out)
        
        return quantized_out, indices, quantizer_loss
    
    def decode(self, x: Tensor, observations, inference=True):
        x = x.flatten(start_dim=-2)
        p_lambda = self.decoder(x)
        
        return (
            self.dOptimizer(observations, p_lambda) if not inference else
            self.dOptimizer.inference(observations, p_lambda)
        )

    def init_jax_functions(self):
        init_model = self.dOptimizer.initializer_model
        init_model_jax = InitializerModuleJax(self.mean, self.std, self.init_shape, init_model.cpu())
        self.dOptimizerJax = DOptimizerJax(init_model_jax, self.num_batch)

    def decode_jax(self, x, observations):
        x = x.flatten(start_dim=-2)
        p_lambda = self.decoder(x)

        return self.dOptimizerJax.inference(observations, p_lambda.cpu().numpy())

    def forward(self, target, observations, inference=False):
        quantized_out, indices, quantizer_loss = self.encode(target)
        return self.decode(quantized_out, observations, inference=inference), quantizer_loss
    
    def compute_loss(self, observations, target, opt_output, quantizer_loss, step, beta):
        trajs, primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch, v_init = opt_output

        primal_sol_x = primal_sol_level_1[:, 0:self.dOptimizer.nvar]
        primal_sol_y = primal_sol_level_1[:, self.dOptimizer.nvar:2*self.dOptimizer.nvar]
        x = (self.dOptimizer.P @ primal_sol_x.T).T
        y = (self.dOptimizer.P @ primal_sol_y.T).T
        trajs = th.hstack([x, y])
        
        rec_loss = th.nn.functional.mse_loss(trajs, target, reduction="mean")
        
        loss = 1e-2 * rec_loss + quantizer_loss 
        return loss, rec_loss, quantizer_loss
    
    def compute_mlp_loss(self, observations, qp2_output):
        trajs, primal_sol_level_1, primal_sol_level_2, accumulated_res, res_norm_batch, v_init = qp2_output

        primal_sol_x = primal_sol_level_2[:, 0:self.dOptimizer.nvar]
        primal_sol_y = primal_sol_level_2[:, self.dOptimizer.nvar:2*self.dOptimizer.nvar]
        x = th.mm(self.dOptimizer.P, primal_sol_x.T).T
        y = th.mm(self.dOptimizer.P, primal_sol_y.T).T
        
        xdot = th.mm(self.dOptimizer.Pdot, primal_sol_x.T).T
        ydot = th.mm(self.dOptimizer.Pdot, primal_sol_y.T).T

        vel_pen = th.linalg.norm(x - self.dOptimizer.v_des*self.dOptimizer.t_fin, dim=1)
        
        heading_angle = th.atan2(ydot, xdot)
        heading_penalty = th.linalg.norm(
            th.maximum(
                th.zeros((self.dOptimizer.num_batch, self.dOptimizer.num), device=self.device),
                th.abs(heading_angle)-10*th.pi/180
            ),
            dim=1
        )
        pro_pen = th.nn.functional.mse_loss(primal_sol_level_2, primal_sol_level_1)

        # k = dict(res=1e0, vel=1e-3, hed=1e-1, cur=1e+1, smo=1e+0, pro=1e-3, cen=1e-9)
        k = dict(res=1e0, vel=1e-1, hed=1e-1, cur=1e+1, smo=1e+0, pro=1e0, cen=1e-9)
        aug_loss = (
            + k["res"] * accumulated_res
            # + k["vel"] * vel_pen
            + k["pro"] * pro_pen
            # + k["hed"] * heading_penalty
            # + k["cen"] * cen_closest
        )
        aug_loss = th.mean(aug_loss)
        return aug_loss


class VQVAELinear(VQVAEModule):
    def __init__(
            self, inp_dim: int, out_dim: int,
            num_embeddings: int, embedding_dim: int, num_features: int,
            encoder_shapes: List[int], decoder_shapes: List[int], beta=0.2, device="cpu", num_batch=1_000
    ):
        super(VQVAELinear, self).__init__(
            out_dim, num_embeddings, embedding_dim,
            num_features, decoder_shapes, beta, device, num_batch
        )
        encoder_shapes = [inp_dim] + encoder_shapes + [num_features*embedding_dim]
        self.encoder = th.nn.Sequential(*[
            th.nn.Sequential(
                th.nn.Linear(in_fea, out_fea),
                th.nn.BatchNorm1d(out_fea),
                th.nn.Tanh() if i-1 == len(encoder_shapes) else th.nn.ReLU()
            )
            for i, (in_fea, out_fea) in enumerate(th.tensor(encoder_shapes).unfold(0, 2, 1))
        ])
        self.to(device=device)
    
    
class VQVAECNN(VQVAEModule):
    def __init__(
            self, mean, std, out_dim: int, 
            hid_channels: int, out_channels: int, kernel_size: int, padding: int,
            num_layers: int, num_embeddings: int, embedding_dim: int, decoder_shapes: List[int],
            init_in_dim: int, init_out_dim: int, init_shape: List[int], beta=0.2, device="cpu", num_batch=1_000
    ):
        super(VQVAECNN, self).__init__(
            mean, std, out_dim, num_embeddings, embedding_dim,
            out_channels, decoder_shapes, init_in_dim,
            init_out_dim, init_shape, beta, device, num_batch
        )
        self.encoder = th.nn.Sequential(
            *[     # input layer
                th.nn.Sequential(
                    th.nn.Conv1d(2, hid_channels, kernel_size, padding=padding),
                    th.nn.BatchNorm1d(hid_channels),
                    th.nn.ReLU()
                )
            ] + [  # hidden layer
                th.nn.Sequential(
                    th.nn.Conv1d(hid_channels, hid_channels, kernel_size, padding=padding),
                    th.nn.BatchNorm1d(hid_channels),
                    th.nn.ReLU()
                )
                for _ in range(num_layers)
            ] + [  # output layer
                th.nn.Sequential(
                    th.nn.Conv1d(hid_channels, out_channels, kernel_size, padding=padding),
                    th.nn.BatchNorm1d(out_channels),
                    th.nn.ReLU(),
                    
                    th.nn.Linear(100, embedding_dim)
                )
            ]
        )
        self.to(device=device)
        
    def encode(self, target):
        target = target.reshape(*target.size()[:-1], 2, 100)
        enc_out = self.encoder(target)
        quantized_out, indices, quantizer_loss = self.quantizer(enc_out)
    
        return quantized_out, indices, quantizer_loss
    
    def forward(self, target, observations):

        quantized_out, indices, quantizer_loss = self.encode(target)
        return super(VQVAECNN, self).decode(quantized_out, observations, inference=False), quantizer_loss
    

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
    
    m = 3
    vqvae_cnn = VQVAECNN(
        1, 1, 30,  10, 7, 11, 5,
        2, 5, 5, [256, 1024, 1024, 1024], 77, 44, [256, 1024, 1024, 1024],
        num_batch=m, device="cpu"
    )
    summary(vqvae_cnn, [(m, 200), (m, 55)], device="cpu", mode="train")
    
    # vqvae_lin = VQVAELinear(
    #     200, 30, 5,  5, 10,
    #     [1024, 1024, 1024, 1024], [256, 1024, 1024, 1024, 1024],
    #     num_batch=1, device="cuda"
    # )
    # summary(vqvae_lin, [(1, 200), (1, 55)], device="cpu")
