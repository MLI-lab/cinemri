
import torch.nn as nn
import torchkbnufft as tkbn
from .transforms import *

class ForwardOperator(nn.Module):
    def __init__(self, dataset_type="cartesian", Nx=-1, Ny=-1):
        super().__init__()
        
        self.dataset_type = dataset_type

        if dataset_type == "non_cartesian":
            self.nufft_ob = tkbn.KbNufft(im_size=(Ny, Nx))
    
    def forward(self, img, sample, smaps=None):
        if smaps is None:
            smaps = sample["smaps"]

        if self.dataset_type == "cartesian":
            return self.forward_cartesian(img, smaps, sample["mask"])
        elif self.dataset_type == "sparse_cartesian":
            return self.forward_sparse_cartesian(img, smaps, sample["mask"])
        elif self.dataset_type == "non_cartesian":
            return self.forward_non_cartesian(img, smaps, sample["trajectory"])
        else:
            raise Exception
        
    def forward_cartesian(self, img, smaps, mask):
        """
        Single slice:
        Input tensor dimensions: (Ns: batch size)
          img: (Ns, Ny, Nx, 2)
          smaps: (Nc, Ny, Nx, 2)
          mask: (Ns, Ny, Nx)
        Output:
          kspace: (Ns, Nc, Ny, Nx, 2)

        2D multi-slice:
        Input tensor dimensions: (Ns: batch size)
          img: (Ns, Nz, Ny, Nx, 2)
          smaps: (Nc, Nz, Ny, Nx, 2)
          mask: (Ns, Nz, Ny, Nx)
        Output:
          kspace: (Ns, Nc, Nz, Ny, Nx, 2)
        """
        img_s = complex_mul(img.unsqueeze(dim=1), smaps.unsqueeze(dim=0))
        kspace_hat = fft2(img_s) # dimensions: (Ns, Nc, Ny, Nx, 2)
        return kspace_hat * mask.unsqueeze(dim=1).unsqueeze(dim=4) # new mask dim: (Ns, 1, Ny, Nx, 1)
    
    def forward_sparse_cartesian(self, img, smaps, mask):
        """
        Single slice:
        Input tensor dimensions: (Ns: batch size)
          img: (Ns, Ny, Nx, 2)
          smaps: (Nc, Ny, Nx, 2)
          mask: (Ns, Nl, Nr, 3)
        Output:
          kspace: (Ns, Nc, Nl, Nr, 2)

        2D multi-slice:
        Input tensor dimensions: (Ns: batch size)
          img: (Ns, Nz, Ny, Nx, 2)
          smaps: (Nc, Nz, Ny, Nx, 2)
          mask: (Ns, Nl, Nr, 3)
        Output:
          kspace: (Ns, Nc, Nl, Nr, 2)
        """

        assert img.ndim == 4 or img.ndim == 5

        Ns, Nl, Nr, _ = mask.shape
        Nc = smaps.shape[0]

        img_s = complex_mul(img.unsqueeze(dim=1), smaps.unsqueeze(dim=0))
        kspace_hat = fft2(img_s) # dimensions: (Ns, Nc, Ny, Nx, 2)

        # shape of mask: (Ns, Nl, Nr, 3)
        mask_flattened = mask.flatten(end_dim=-2)

        # extract masked coordinates from the k-space matrix
        if img.ndim == 4:
            return kspace_hat.flatten(end_dim=1)[:, mask_flattened[:,1], mask_flattened[:,2], :].reshape((Ns, Nc, Nl, Nr, 2))
        else:
            return kspace_hat.flatten(end_dim=1)[:, mask_flattened[:,0], mask_flattened[:,1], mask_flattened[:,2], :].reshape((Ns, Nc, Nl, Nr, 2))
        
    def forward_non_cartesian(self, img, smaps, trajectory):
        """
        Single slice:
        Input tensor dimensions: (Ns: batch size)
          img: (Ns=1, Ny, Nx, 2) float32
          smaps: (Nc, Ny, Nx, 2) float32
          trajectory: (Ns=1, Nl, Nr, 3) float32
        Output:
          kspace: (Ns=1, Nc, Nl, 2) float32
        """

        assert img.shape[0] == 1, "not implemented/tested yet"
        assert img.ndim == 4

        Nc = smaps.shape[0]
        _, Nl, Nr, _ = trajectory.shape

        img_smaps = complex_mul(img.unsqueeze(dim=1), smaps.unsqueeze(dim=0))
        trajectory_flattened = trajectory.flatten(end_dim=-2)[:, 1:3].T # new shape: (2, Nl*Nr)
        kspace_rec = self.nufft_ob(img_smaps, trajectory_flattened) # (Ns=1, Nc, Nl*Nr, 2) float32

        return kspace_rec.reshape((1, Nc, Nl, Nr, 2))

        





