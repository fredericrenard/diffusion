from torch import nn
import torch
class DenoisingDiffusionModel(nn.Module):
    
    def __init__(self,
                 unet_model: nn.Module,
                 timesteps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,):
        
        super().__init__()
        self.unet = unet_model
        self.alpha_start = 1 - beta_start
        self.betas_t = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas_t = 1 - self.betas_t
        self.alpha_bar_t = torch.cumprod(self.alphas_t, dim=0)
        self.sqrt_alphas_t = torch.sqrt(self.alpha_bar_t)
        self.sqrt_alpha_bar_t = torch.sqrt(self.alpha_bar_t)
        self.sqrt_1_minus_alpha_bar_t = torch.sqrt(1 - self.alpha_bar_t)

    def loss(self, x_0: torch.Tensor, t: torch.Tensor):
        """Loss function for the denoising diffusion model."""
        eps = torch.randn_like(x_0)
        x_t = self.sqrt_alpha_bar_t[t] * x_0 + self.sqrt_1_minus_alpha_bar_t[t] * eps
        eps_theta = self.unet(x_t, t)
        loss = nn.functional.mse_loss(eps, eps_theta)
        return loss
    
    def sample(self, n_samples: int, shape: tuple[int, ...]):
        """Sample from the denoising diffusion model."""
        x_t = torch.randn(n_samples, *shape)
        
        for t in range(self.timesteps - 1, -1, -1):
            if t > 1:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)
                
            x_t = (1 / self.sqrt_alphas_t[t]) * (x_t - ((1 - self.alphas_t[t]) / self.sqrt_1_minus_alpha_bar_t[t]) * self.unet(x_t, t))
        return x_t


#### Sampling