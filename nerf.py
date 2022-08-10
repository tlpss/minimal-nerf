from dis import dis
import torch.nn as nn
import torch
import pytorch_lightning as pl
import numpy as np
import torch.random 

class NeRFNetwork(nn.Module):
    def __init__(self, depth=8, width=256, skips=[4]):
        super().__init__()
        self.depth = depth
        self.width = width 
        self.skips = skips

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.encoding = nn.ModuleList([nn.Linear(3,self.width)])
        for i in range(self.depth-1):
            depth = i +1
            if depth in self.skips:
                self.encoding.append(nn.Linear(self.width+3,self.width))
            else:
                self.encoding.append(nn.Linear(self.width,self.width))

        self.density_head = nn.Linear(self.width,1)
        self.encoding_head = nn.Linear(self.width,128)
        self.rgb_head = nn.Linear(128+3,3)

    def forward(self,x):

        position, view_direction = torch.split(x,3,dim=1)
        print(position)
        x = position
        for i,layer in enumerate(self.encoding):
            if i in self.skips:
                x = torch.cat([x,position],dim=1)
            x = layer(x)
            x = self.relu(x)

        density = self.density_head(x)
        density = self.relu(density) # constrain to 0- Inf.

        features = self.encoding_head(x)
        features = self.relu(features)

        x = torch.cat([features, view_direction],dim=1)
        rgb = self.rgb_head(x)
        rgb = self.sigmoid(rgb) # constrain to 0-1


        return rgb, density
   
class NeRF(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()


        #todo: get these from colmap?
        self.min_render_distance = 0
        self.max_render_distance = 1.0

        self.n_coarse_samples = 5 
        self.nerf = NeRFNetwork()

    def forward(self, x):
        return self.nerf(x)

    
    def render_view(self,camera_pose, hight, width,intrinsics_matrix, n_coarse_samples):
        pass

    def render_rays(self,ray_origins, ray_directions, n_coarse_samples):

        network_query_points = self._compute_query_points(ray_origins,ray_directions,n_coarse_samples)
        network_queries = torch.cat([network_query_points, ray_directions],dim=1)
        queried_rgbs, queried_densities = self.forward(network_queries)

        rendered_rgbs = self._volume_render_rays(network_query_points,queried_rgbs, queried_densities)

        return rendered_rgbs 

    def _compute_query_points(self,ray_origins, ray_directions, n_coarse_samples):
        """

        Args:
            ray_origins (_type_): _description_
            ray_directions (_type_): _description_
            n_coarse_samples (_type_): _description_

        Returns:
            n_rays x n_coarse_samples x 3
        """
        bins = torch.linspace(self.min_render_distance, self.max_render_distance,steps =n_coarse_samples +1)
        bin_width = (self.max_render_distance-self.min_render_distance)/n_coarse_samples
        uniform_bin_samples = torch.rand(ray_origins.shape[0],n_coarse_samples) 
        uniform_bin_samples *= bin_width
        samples = bins[:-1] + uniform_bin_samples

        # todo: fix that this results in Nrays x Nsamples x 3 output...
        query_points = ray_origins + ray_directions[...,None] * samples
        return query_points

    def _volume_render_rays(self, queried_points, queried_rgbs, queried_densities):
        
        distances = self.compute_adjacent_distances(queried_points)
        neg_weighted_densities = - distances * queried_densities
        accumulated_transmittances = torch.exp(torch.cumprod(neg_weighted_densities,dim=1))

        weights = (1- torch.exp(-distances * queried_densities)) * accumulated_transmittances

        rgbs = torch.sum(weights * queried_rgbs, dim = -2)
        return rgbs
        

    @staticmethod
    def compute_adjacent_distances(query_points: torch.Tensor) -> torch.Tensor:
        """computes adjacent value distances for each sample. Adds epsilon distance for the last sample.

        Args:
            query_points (_type_): N_rays x N_samples x 3 tensor

        Returns:
            distances N_rays x N_samples x 1 tensor
        """
        distances = query_points[..., 1:] - query_points[..., :-1]
        eps_distances_for_final_sample = torch.ones_like(query_points[...,:1])*1e-10
        distances = torch.cat([distances,eps_distances_for_final_sample], dim =-1)
        return distances

    def training_step(self,batch,batch_idx):
        ray_origins, ray_directions,rgb_targets = batch

        rgb_values, *_ = self.render_rays(ray_origins, ray_directions,self.n_coarse_samples)

        loss = self.mse(rgb_values,rgb_targets)
        self.log("loss",loss)
        return loss 


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=4e-4)


if __name__ == "__main__":
    # nerf = NeRF()
    #nerf.render_rays(np.zeros((3,3)),None,10)

    nerf = NeRF()
    batch = torch.rand((4,3)), torch.rand((4,3)), torch.rand((4,3))
    y=nerf.training_step(batch,0)
    print(y)