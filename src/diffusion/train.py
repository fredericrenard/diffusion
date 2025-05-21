import torch
from diffusion.unet import UNet
import torchvision
from diffusion.model import DenoisingDiffusionModel
from tqdm import tqdm

def train() -> None:
    
    data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    
    timesteps = 1000
    n_epochs = 3
    batch_size = 128
    learning_rate = 1e-4
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    
    model = DenoisingDiffusionModel(unet_model=UNet(timesteps=timesteps, 
                                                    time_embedding_dim=128,
                                                    in_channels=1,
                                                    base_dim=128,
                                                    dim_mults=[2,4]), 
                                                    beta_start=0.0001, 
                                                    beta_end=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in tqdm(range(n_epochs)):
        for batch in dataloader:
            x_0 = batch[0] # [B, 1, 28, 28]
            t = torch.randint(1, timesteps, (x_0.shape[0],)) # [B]
            loss = model.loss(x_0, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")


if __name__ == "__main__":
    train()