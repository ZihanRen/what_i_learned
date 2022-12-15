#%%
# component:
#(1) Progressive GAN. You need to be slow at the beginning if you want to grow.
#(2) z --> w; inject w vector to generator at different stages; noise mapping network
#(3) adaptive instance normalization (AdaIN)

#advantage of using styleGAN
#(1) Greater fidelity; better control over features; style is any variation in the image
#(2) z-->w mapping network help to solve the problem of entanglement; help distangle features for GAN training
# mapping network help transform normal space into feature density space

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm
def get_truncated_noise(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-truncation,truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)


class MappingLayers(nn.Module): 
    def __init__(self, z_dim, hidden_dim, w_dim):
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(z_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,w_dim)
        )

    def forward(self, noise):
        return self.mapping(noise)


class InjectNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter( 
            data = torch.randn(1,channels,1,1)
        )

    def forward(self, image):
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])        
        noise = torch.randn(noise_shape, device=image.device)
        return image + self.weight * noise 

class AdaIN(nn.Module):
    # inject style to image
    def __init__(self, channels, w_dim):
        super().__init__()

        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        normalized_image = self.instance_norm(image)
        # w vector is injected into image (fake) for each channel
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
        transformed_image = style_scale*normalized_image + style_shift
        return transformed_image

class MicroStyleGANGeneratorBlock(nn.Module):
    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample
        # upsample for progress GAN
        if self.use_upsample:
            self.upsample = nn.Upsample((starting_size), mode='bilinear')
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1)
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan,w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.adain(x, w)
        x = self.activation(x)
        return x


class MicroStyleGANGenerator(nn.Module):

    def __init__(self, 
                 z_dim, 
                 map_hidden_dim,
                 w_dim,
                 in_chan,
                 out_chan, 
                 kernel_size, 
                 hidden_chan):
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        # start_size=4, no upscample
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        # upsample 4*4 to 8*8
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        # upscample 8*8 to 16*16
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2

    def upsample_to_match_size(self, smaller_image, bigger_image):
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode='bilinear')

    def forward(self, noise, return_intermediate=False):
        x = self.starting_constant
        w = self.map(noise) # transform noise to w
        x = self.block0(x, w) # style injection Conv layer (4,4)
        x_small = self.block1(x, w) # style injection Conv layer 2 (upsample) (8,8)
        x_small_image = self.block1_to_image(x_small) # reshape layer2 feature map to image channel
        x_big = self.block2(x_small, w)  # another style Conv layer 3
        x_big_image = self.block2_to_image(x_big) # transform layer3 to output image
        # match the size of small image to large image
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image)
        # final output is a combination between two layers output, small and large
        interpolation = self.alpha * (x_big_image) + (1-self.alpha) * (x_small_upsample)
        
        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation


#%%
style_gen = MicroStyleGANGenerator(
    z_dim=20,
    map_hidden_dim=100,
    w_dim=200,
    in_chan=512,
    out_chan=3,
    kernel_size=2,
    hidden_chan=300
    )
from torchinfo import summary
summary(style_gen,(2,20)) 
# %%
