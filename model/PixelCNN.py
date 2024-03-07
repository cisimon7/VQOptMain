import torch as th
from torchinfo import summary


class MaskedConv1d(th.nn.Conv1d):
    MASK_TYPES = ["A", "B"]
    
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)
        assert mask_type in self.MASK_TYPES
        self.mask_type = mask_type

        self.register_buffer('mask', self.weight.data.clone())
        _, _, size = self.weight.size()

        self.mask.fill_(1)
        self.mask[:, :, size // 2 + (mask_type == "B"):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)
    
    
class CondGatedMaskedConv1d(th.nn.Module):
    def __init__(self, *args, **kwargs):
        super(CondGatedMaskedConv1d, self).__init__()
        self.masked_conv_1 = MaskedConv1d(*args, **kwargs)
        self.masked_conv_2 = MaskedConv1d(*args, **kwargs)
        self.cond_conv_1 = th.nn.Conv1d(1, args[2], 1)
        self.cond_conv_2 = th.nn.Conv1d(1, args[2], 1)
        self.tanh = th.nn.Tanh()
        self.sigm = th.nn.Sigmoid()

    def forward(self, x, h):
        inp = self.tanh(self.masked_conv_1(x) + self.cond_conv_1(h))
        gate = self.sigm(self.masked_conv_2(x) + self.cond_conv_2(h))
        return inp * gate
    
    
class PixelCNN(th.nn.Module):
    def __init__(
            self, num_embedding, kernel_size, padding=1, in_channels=1, n_channels=32,
            n_layers=7, device="cpu"
    ):
        super(PixelCNN, self).__init__()
        self.layers = th.nn.ModuleList()
        
        self.layers.append(
            CondGatedMaskedConv1d('A', in_channels, n_channels, kernel_size, 1, padding, bias=False)
        )
        self.layers.append(th.nn.BatchNorm1d(n_channels))

        for _ in range(1, n_layers+1):
            self.layers.append(
                CondGatedMaskedConv1d('B', n_channels, n_channels, kernel_size, 1, padding, bias=False)
            )
            self.layers.append(th.nn.BatchNorm1d(n_channels))

        self.layers.append(th.nn.Conv1d(n_channels, num_embedding, 1))
        self.to(device=device)

    def forward(self, x, h):
        out = x
        for layer in self.layers:
            if isinstance(layer, CondGatedMaskedConv1d):
                out = layer(out, h)
            else:
                out = layer(out)
        return out
    

class PixObsNet(th.nn.Module):
    def __init__(self, mean, std, layers, input_shape=55, output_shape=11, device="cpu"):
        super().__init__()
        self.std = std
        self.mean = mean
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        layers = [input_shape] + layers + [output_shape]
        self.net = th.nn.Sequential(*[
            th.nn.Sequential(
                th.nn.Linear(in_fea, out_fea),
                th.nn.BatchNorm1d(out_fea),
                th.nn.Tanh() if i-1 == len(layers) else th.nn.ReLU()
            )
            for i, (in_fea, out_fea) in enumerate(th.tensor(layers).unfold(0, 2, 1))
        ])
        self.to(device=device)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.net(x).unsqueeze(dim=-2)
    

class PixObsNetCNN(th.nn.Module):
    def __init__(self, mean, std, hid_channels, kernel_size, padding, num_layers, out_channels=1, output_shape=11, device="cpu"):
        super().__init__()
        self.std = std
        self.mean = mean
        self.output_shape = output_shape

        self.net = th.nn.Sequential(
            *[     # input layer
                th.nn.Sequential(
                    th.nn.Conv1d(5, hid_channels, kernel_size, padding=padding),
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

                    th.nn.Linear(11, output_shape), 
                )
            ]
        )
        self.to(device=device)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = th.stack(x.chunk(5, -1), dim=-1).mT
        return self.net(x)
    
    
if __name__ == "__main__":
    pixel_cnn = PixelCNN(20, 3, 1)
    summary(pixel_cnn, [(1, 1, 7), (1, 1, 7)], device="cuda")

    pixel_obs = PixObsNet(0, 0, [256, 1024, 1024])
    summary(pixel_obs, [(1, 55)], device="cuda")
    
    pixel_obs_cnn = PixObsNetCNN(0, 0, 10, 11, 5, 3)
    summary(pixel_obs_cnn, [(1, 55)], device="cuda")
    