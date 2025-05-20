import torch
from diffusion.model import DenoisingDiffusionModel
from diffusion.unet import UNet
import matplotlib.pyplot as plt
def inference() -> None:
    timesteps = 1000
    n_samples = 10
    model = DenoisingDiffusionModel(unet_model=UNet(timesteps=timesteps, 
                                                    time_embedding_dim=128,
                                                    in_channels=1,
                                                    base_dim=128,
                                                    dim_mults=[2,4]), 
                                                    beta_start=0.0001, 
                                                    beta_end=0.02)
    model.load_state_dict(torch.load("model.pth"))
    
    for i in range(n_samples):
        x_t = model.sample(1, (1, 28, 28))
        plt.imshow(x_t.squeeze().cpu().numpy())
        plt.show()
        
if __name__ == "__main__":
    inference()