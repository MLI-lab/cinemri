
import torch.nn as nn
import torchkbnufft as tkbn
from .transforms import *

class ForwardOperator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img, sample, smaps=None):
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

        if smaps is None:
            smaps = sample["smaps"]
        mask = sample["mask"]

        assert img.ndim == 4 or img.ndim == 5

        Ns, Nl, Nr, _ = mask.shape
        Nc = smaps.shape[0]

        img_s = complex_mul(img.unsqueeze(dim=1), smaps.unsqueeze(dim=0))
        kspace_hat = fft2(img_s) # dimensions: (Ns, Nc, Ny, Nx, 2)

        # shape of mask: (Ns, Nl, Nr, 3)
        mask_flattened = mask.flatten(end_dim=-2)

        # shape of mask: (Ns, Nl, Nr, 3)
        batch_sample_indices = torch.arange(Ns, device=kspace_hat.device).reshape(Ns, 1, 1).repeat(1, Nl, Nr).flatten()

        # indices = torch.concat((mask_flattened, batch_sample_indices), dim=-1)

        if img.ndim == 4:
            return kspace_hat[batch_sample_indices, :, mask_flattened[:,1], mask_flattened[:,2], :].reshape((Ns, Nl, Nr, Nc, 2)).permute((0, 3, 1, 2, 4))
        else:
            return kspace_hat[batch_sample_indices, :, mask_flattened[:,0], mask_flattened[:,1], mask_flattened[:,2], :].reshape((Ns, Nl, Nr, Nc, 2)).permute((0, 3, 1, 2, 4))

        





