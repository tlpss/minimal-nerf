import torch 
from torch.utils.data import Dataset
import numpy as np
from nerf import get_image_rays

class TinyNerfDataset(Dataset):
    def __init__(self,path = 'minimal-nerf/tiny_nerf_data.npz') -> None:
        super().__init__()
        data = np.load(path)
        self.images = data['images']
        self.poses = torch.tensor(data['poses'])
        self.focal = data['focal']
        self.H, self.W = self.images.shape[1:3]
    
        # 100 imgs x 600x600 pixels -> dataset size = 100x600**2*(3+3+3) ~0.3GB (CPU RAM) -> precompute..

        self.ray_origins, self.ray_directions = [], []
        for pose in self.poses:
            ray_origins, ray_directions = get_image_rays(pose,self.H,self.W,self.focal)
            self.ray_origins.append(ray_origins)
            self.ray_directions.append(ray_directions)
        
        self.ray_origins = torch.cat(self.ray_origins).flatten(end_dim=-2)
        self.ray_directions = torch.cat(self.ray_directions).flatten(end_dim=-2)
        self.pixels = torch.tensor(self.images).flatten(end_dim=-2)

        assert self.ray_origins.shape == (len(self.images)*self.H*self.W,3)
        assert self.ray_directions.shape == self.pixels.shape

    def __getitem__(self, idx):
        
        # get the rays for all items
        # return ray origins, ray 
        return self.ray_origins[idx],self.ray_directions[idx],self.pixels[idx]

    def __len__(self):
        return len(self.pixels)

    def getimage(self,idx):
        assert idx < len(self.images)
        return self.images[idx]


if __name__ == "__main__":
    dataset = TinyNerfDataset()

