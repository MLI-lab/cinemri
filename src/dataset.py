import scipy
import scipy.io
from typing import Iterable
import torch
import numpy as np
from .transforms import *
from .helpers import *

class Dataset():
    """ 
        This class stores a Cartesian dataset without zeros in the k-space and stores a tensor of measured k-space coordinates (trajectory) instead of a sampling mask. 
        
        attributes:
        - `kspace`: shape (Nl, Nc, Nr, 2) contains the complex k-space measurements (real, imaginary part in last dimension)
        - `trajectory`: shape (Nl, Nr, 3) contains the k-space coordinates of the measurements normalized to 1/FOV. Order of the dimensions: z y x.
        - `mask`: shape (Nl, Nr, 3) contains the indices of the measured coordinates on the Cartesian grid (same information as in trajectory, but in a more accessible format). Order of the dimensions: z y x.
        - `smaps`:  shape (Nc, Nz, Ny, Nx, 2) torch.tensor float32
        - `transform`: optional function: dict -> dict

        naming:
        - `Nk`: number of frames
        - `Nl`: number of measured k-space lines
        - `Nr`: number if measurements in the read-out direction (x-direction)
        - `Nc`: number of receiver coils
        - `Nz`, `Ny`, `Nx`: resolution in z, y, and x-direction
        """

    def __init__(self, kspace, trajectory, mask, line_indices, smaps, transform=None):
        self.kspace = kspace # shape (Nk, Nc, Nl, Nr, 2) torch.tensor float32
        self.trajectory = trajectory # shape (Nk, Nl, Nr, 3) torch.tensor float32
        self.mask = mask # shape (Nk, Nl, Nr, 3) LongTensor
        self.line_indices = line_indices # shape (Nk, Nl) LongTensor
        self.smaps = smaps # shape (Nc, Nz, Ny, Nx, 2) torch.tensor float32
        self.transform = transform # function: transform(dict: sample) -> dict

        assert kspace.ndim == 5
        assert trajectory.ndim == 4
        assert mask.ndim == 4
        assert line_indices.ndim == 2
        assert smaps.ndim == 5
        assert kspace.shape[-1] == 2
        assert trajectory.shape[-1] == 3
        assert mask.shape[-1] == 3

        self.Nk, self.Nc, self.Nl, self.Nr, _ = kspace.shape
        _, self.Nz, self.Ny, self.Nx, _ = smaps.shape

    @classmethod
    def load_from_matfile(self, matfile_path,
                                transform=None,
                                remove_padding=False,
                                set_smaps_outside_to_one=False,
                                validation_percentage=0.,
                                number_of_lines_per_frame=6,
                                max_Nk=-1,
                                seed=1998):
        """ 
        Loads `matfile_path` that stores measurement data.
        Randomly extracts `validation_percentage` percent of the k-space lines for validation.
        Bins the remaining lines into frames with `number_of_lines_per_frame` lines each. If `max_Nk` is specified, the number of frames is reduced and excess frames are discarded.


        Arguments:
        - `transform`: An optional function that is applied to every sample data loaded from the dataset with the get_item() method.
        - `remove_padding`: By default, the k-space data of the scanner is zero-padded and the smaps are computed on a larger grid. If `remove_padding` is true, the padding is removed and the smaps are cropped in the Fourier domain.
        - `set_smaps_outside_to_one`: By default, the smaps estimated by the scanner have zero entries outside the human body (they cannot be estimated outside, as there is no signal). Thus, the reconstructions can take arbitrary values outside the body without affecting the reconstruction loss. If `set_smaps_outside_to_one` is true, the zero-sensitivities are set to 1.0. By setting the smaps outside to 1.0, the reconstructions are forced to zero outside the human body.
        - `validation_percentage`: Percentage of the k-space lines that are used for validation. Default: 0.
        - `number_of_lines_per_frame`: Number of k-space lines per frame. Default: 6.
        - `max_Nk`: If not -1, the number of frames is limited.
        - `seed`: seed for the random extraction of k-space lines.

        Outputs: (training_dataset, validation_dataset)
        - training_dataset: A dataset with `number_of_lines_per_frame` lines per frame.
        - validation_dataset: A dataset with 1 line per frame.
        """

        # save the current state of the RNG and set the new one
        random_state = np.random.get_state() 
        np.random.seed(seed)

        mat = scipy.io.loadmat(matfile_path)

        raw_smaps = mat["smaps"].squeeze()
        Nc, Nyold, Nxold, _ = raw_smaps.shape

        # if required, set the zero pixels of the smaps (outside the body) to 1.0
        if set_smaps_outside_to_one:
            raw_smaps[raw_smaps == 0.] = 1.


        # if required, remove the zero-padding from the k-space and truncate the smaps in the Fourier domain
        if remove_padding: # remove the padding

            # compute the resolution in x-direction without padding
            Nx = int(mat["kx_range"][0,1] - mat["kx_range"][0,0] + 1)
            
            # compute the resolution in y-direction without padding
            if -mat["ky_range"][0,0] == mat["ky_range"][0,1] + 1: # standard case
                Ny = int(mat["meta"]["ky_range"][0,1] - mat["meta"]["ky_range"][0,0] + 1)
            elif -mat["ky_range"][0,0] < mat["ky_range"][0,1]: # probably partial-Fourier
                Ny = int(2 * mat["ky_range"][0,1])
            else: # unknown case
                print(mat["ky_range"], raw_smaps.shape[2])
                raise Exception

            # truncate the smaps in the Fourier domain
            Nystart, Nxstart = int((Nyold - Ny) / 2), int((Nxold - Nx) / 2)

            smaps = torch.tensor(raw_smaps)
            smaps_fft = fft2(smaps)
            smaps_fft = smaps_fft[:,Nystart:(Nystart+Ny), Nxstart:(Nxstart+Nx), :]
            smaps = ifft2(smaps_fft)

        else: # keep the padding
            Nc, Ny, Nx, _ = raw_smaps.shape
            smaps = torch.tensor(raw_smaps)


        Nl, _, Nr, _ = mat["kspace"].shape
        
        # generate a random subset of measured lines:
        validation_indices = np.arange(stop=Nl)
        np.random.shuffle(validation_indices)
        validation_indices = validation_indices[0:int(Nl * validation_percentage / 100)]

        Nk = (Nl - len(validation_indices)) // number_of_lines_per_frame
        if max_Nk != -1:
            Nk = min(Nk, max_Nk)

        # count the number of validation lines within Nk
        j = 0
        Nk_validation = 0
        for k in range(Nk):
            for l in range(number_of_lines_per_frame):
                while j in validation_indices:
                    Nk_validation += 1
                    j += 1
                j += 1

        
        # create tenors for the training and the validation dataset
        kspace = np.zeros((Nk, Nc, number_of_lines_per_frame, Nr, 2), dtype=np.float32)
        mask = np.zeros((Nk, number_of_lines_per_frame, Nr, 3), dtype=np.int64)
        trajectory = np.zeros((Nk, number_of_lines_per_frame, Nr, 3), dtype=np.float32)
        line_indices = np.zeros((Nk, number_of_lines_per_frame), dtype=np.int64)

        kspace_validation = np.zeros((Nk_validation, Nc, 1, Nr, 2), dtype=np.float32)
        mask_validation = np.zeros((Nk_validation, 1, Nr, 3), dtype=np.int64)
        trajectory_validation = np.zeros((Nk_validation, 1, Nr, 3), dtype=np.float32)
        line_indices_validation = np.zeros((Nk_validation, 1), dtype=np.int64)

        # find the zero index of the k-space matrix in ky-direction
        ky_zero_index = int(Ny / 2)
        ky_zero_index_with_padding = int(Nyold / 2)
        # find the first kx index that should be filled with data
        kx_shift = int(Nx / 2 - Nr / 2)
        kx_shift_with_padding = int(Nxold / 2 - Nr / 2)

        

        # stacks the measurements from different coils
        def line_generator():
            for l in range(Nl):
                line_kspace = mat["kspace"][l,:,:,:]

                ky = mat["mask"][l,:,1] - ky_zero_index_with_padding
                ky_index = ky_zero_index + ky
                ky_coordinate = 2. * np.pi * ky / Ny

                kx = mat["mask"][l,:,2] - kx_shift_with_padding
                kx_indices = kx + kx_shift
                kx_coordinates = 2. * np.pi * (-int(Nr/2) + np.arange(Nr)) / Nx

                line_mask = np.stack((np.zeros(Nr), np.ones(Nr)*ky_index, kx_indices), axis=-1)
                line_trajectory = np.stack((np.zeros(Nr), np.ones(Nr)*ky_coordinate, kx_coordinates), axis=-1)

                yield line_kspace, line_trajectory, line_mask
        lines = line_generator()

        # copy lines to new dataset
        j = 0
        k_validation = 0
        for k in range(Nk):
            for l in range(number_of_lines_per_frame):
                while j in validation_indices: # put lines in the validation dataset
                    line_kspace, line_trajectory, line_mask = next(lines)

                    kspace_validation[k_validation, :, 0, :, :] = line_kspace
                    trajectory_validation[k_validation, 0, :, :] = line_trajectory
                    mask_validation[k_validation, 0, :, :] = line_mask
                    line_indices_validation[k_validation, 0] = j

                    k_validation += 1
                    j += 1
                
                # put the line in the training dataset
                line_kspace, line_trajectory, line_mask = next(lines)
                kspace[k, :, l, :, :] = line_kspace
                trajectory[k, l, :, :] = line_trajectory
                mask[k, l, :, :] = line_mask
                line_indices[k, l] = j

                j += 1

    
        # insert dimensions for the z-axis (that is not used since this method handles 2D datasets with a single slice)
        smaps = smaps.unsqueeze(dim=1)

        line_indices = to_tensor(line_indices)
        kspace = to_tensor(kspace)
        trajectory = to_tensor(trajectory)
        mask = to_tensor(mask)

        line_indices_validation = to_tensor(line_indices_validation)
        kspace_validation = to_tensor(kspace_validation)
        trajectory_validation = to_tensor(trajectory_validation)
        mask_validation = to_tensor(mask_validation)

        # restore the state of the RNG
        np.random.set_state(random_state)
        return self(kspace, trajectory, mask, line_indices, smaps, transform=transform), self(kspace_validation, trajectory_validation, mask_validation, line_indices_validation, smaps, transform=transform)
    
    def shape(self):
        """
        Returns (Nk, Nc, Nz, Ny, Nx).
        """
        return (self.Nk, self.Nc, self.Nz, self.Ny, self.Nx)
        
    def subset(self, subset_indices):
        return Subset(self, subset_indices)

    def __len__(self):
        return self.Nk

    def __getitem__(self, k):        

        if isinstance(k, int):
            indices = [k]
        elif isinstance(k, slice):
            indices = range(k.start or 0, k.stop or self.Nk, k.step or 1)
        elif isinstance(k, Iterable):
            indices = k
        else:
            raise Exception("invalid index format")

        sample = {
            'indices': torch.tensor(indices, dtype=torch.int64),
            'kspace': self.kspace[indices, :, :, :, :],
            'trajectory': self.trajectory[indices, :, :, :],
            'mask': self.mask[indices, :, :, :],
            'line_indices': self.line_indices[indices, :],
            'smaps': self.smaps
        }

        if(self.transform):
            return self.transform(sample)
        return sample




class Subset():
    """
    Helper class that indices a subset of frames in `CartesianDataset` datasets.
    
    A subset of elements of dataset is indexed with indices 0 ... len(subset_indices)-1
    """
    def __init__(self, dataset, subset_indices):
        """
        Returns a new dataset-like object that contains/addresses a subset of samples.
        
        It can be used to only train on a subset of samples and still use the default torch.utils.data.DataLoader.
        
        Example usage:
        ```python
        subset = slice_dataset.subset([0, 2, 4, 6])
        ```
        or
        ```python
        subset = slice_dataset.subset(range(0, 8, 2))
        ```
        or
        ```python
        subset = slice_dataset.subset(slice(0, 8, 2))
        ```
        Then, `subset[1]` returns `slice_dataset[2]`.
        """
        if isinstance(subset_indices, list):
            subset_indices = subset_indices
        elif isinstance(subset_indices, slice):
            subset_indices = list(range(subset_indices))
        elif isinstance(subset_indices, range):
            subset_indices = list(subset_indices)
        else:
            raise Exception("invalid indexing")

        assert max(subset_indices) < dataset.Nk, "Subset indices larger than dataset Nk."
        assert min(subset_indices) >= 0, "Subset indices must be positive integers."

        self.dataset = dataset
        self.subset_indices = subset_indices
        self.Nk = len(subset_indices)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            k, z = idx
            assert isinstance(k, int), "Subset currently only supports indexing by integers." # TODO make this work with slice() indexing and arrays
            return self.dataset[self.subset_indices[k], z]
        else:
            return self.dataset[self.subset_indices[idx]]

    # forward all methods TODO: Can this be implemented more elegantly?
    def __getattr__(self, item):
       return getattr(self.dataset, item)

    def __repr__(self):
       return repr(self.dataset)

    def shape(self):
        if len(self.dataset.shape()) == 4:
            _, Nc, Ny, Nx = self.dataset.shape()
            return (len(self.subset_indices), Nc, Ny, Nx)
        else:
            _, Nc, Nz, Ny, Nx = self.dataset.shape()
            return (len(self.subset_indices), Nc, Nz, Ny, Nx)

    def __len__(self):
        return len(self.subset_indices)


class DatasetCache(Dataset):
    """
    Stores all samples that are requested on the Video-RAM or the RAM. All arrays in the returned samples are on the GPU.
    
    Example usage:
    ```python
    dataset = DatasetCache(CartesianSliceDataset.from_cartesian_dataset(dataset))
    sample = dataset[0]
    ```

    The cache accepts a transform method. If provided, it is applied to the samples *after* loading samples from the cache. Thus, it can be used for elaborate preprocessing. The preprocessed outputs are not stored in the cache.
    """
    def __init__(self, dataset, max_numel_gpu=90, transform=None):
        self.dataset = dataset
        self.cacheData = {} # holds all the data frames indexed by element's index on the CPU
        self.cacheDataGPU = {}
        self.transform = transform
        self.max_numel_gpu = max_numel_gpu

    def apply_transform(self, sample):
        if self.transform is not None:
            return self.transform(sample)
        return sample

    def __getitem__(self, idx):

        if isinstance(idx, list) or isinstance(idx, slice): # make lists and slices hashable
            idx = tuple(idx)

        if idx in self.cacheDataGPU:
            return self.apply_transform(self.cacheDataGPU[idx])

        if idx in self.cacheData:
            return self.apply_transform(self.copyToGPU(self.cacheData[idx]))
        
        sample = self.dataset[idx]
        # cache miss -> load data into cache
        if len(self.cacheDataGPU) < self.max_numel_gpu:
            self.cacheDataGPU[idx] = self.copyToGPU(sample)
        else:
            self.cacheData[idx] = self.copyToCPU(sample)
        
        return self.apply_transform(sample)

    @staticmethod
    def copyToCPU(entry):
        copy = {}
        for k in entry.keys():
            if torch.is_tensor(entry[k]):
                copy[k] = entry[k].to("cpu") # would detach make it faster?
            else:
                copy[k] = entry[k]
        return copy

    @staticmethod
    def copyToGPU(entry):
        copy = {}
        for k in entry.keys():
            if torch.is_tensor(entry[k]):
                copy[k] = entry[k].to("cuda") # does detach make it faster?
            else:
                copy[k] = entry[k]
        return copy
    
    # forward all methods TODO: Can this be implemented more elegantly?
    def __getattr__(self, item):
       return getattr(self.dataset, item)

    def __repr__(self):
       return repr(self.dataset)

    def __len__(self):
        return len(self.dataset)


def collocate_concat(samples):
    """
    Custom collocate function for loading mini-batches using DataLoader

    Similar to the default torch collocate_fn, but concatenates tensors along first dimension instead of stacking.

    Usage:
    ```python
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collocate_concat)
    sample = next(iter(dataloader))
    ```
    """
    elem = samples[0]
    keys = elem.keys()
    batch = {}
    for key in keys:
        if key == "smaps": # we do not need multiple copies of smaps
            batch[key] = elem[key]
        elif key == "shape":
            batch[key] = tuple([len(samples)] + list(elem[key][1:]))
        elif isinstance(elem[key], torch.Tensor):
            batch[key] = torch.cat([e[key] for e in samples], dim=0)
        elif isinstance(elem[key], np.ndarray):
            batch[key] = np.concatenate([e[key] for e in samples], axis=0)
        elif isinstance(elem, int) or isinstance(elem, float):
            batch[key] = torch.tensor([e[key] for e in samples])
        else: # only keep value of first element
            batch[key] = elem[key]
    batch["batch_size"] = len(samples)
    return batch