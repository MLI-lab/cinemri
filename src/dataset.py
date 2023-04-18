import os, re, math
from typing import Iterable
import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from .transforms import *
from .helpers import *


class CartesianDataset():
    """ 
    This is the parent class for all Cartesian datasets with dense data representation. The frequencies between the sampled coordiantes are zero. The implementation is memory inefficient but may be faster than storing the sampled coordiantes in a sparse manner, see `SparseCartesianDataset`. 
    """
    def __init__(self, kspace, mask, smaps=None, reference=None, transform=None, additional_data=None, line_indices=None):

        assert kspace.ndim == 6 and mask.ndim == 4
        if smaps is not None: assert smaps.ndim == 5
        if reference is not None: assert reference.ndim == 4

        self.kspace = kspace # shape (Nk, Nc, Nz, Ny, Nx, 2) torch.tensor float32
        self.mask = mask # shape (Nk, Nz, Ny, Nx) torch.tensor float32
        self.smaps = smaps # shape (Nc, Nz, Ny, Nx, 2) torch.tensor float32
        self.reference = reference # shape (Nk, Nz, Ny, Nx) torch.tensor complex
        self.transform = transform # function: transform(dict: sample) -> dict
        self.additional_data = additional_data # arbitrary data 
        self.line_indices = line_indices

        self.Nk, self.Nc, self.Nz, self.Ny, self.Nx, _ = kspace.shape

    
    @classmethod
    def from_matfile_sense2d(self, matfile_path, transform=None, load_in_chunks=False):
        """
        Loads a SENSE dataset from a .mat file.

        The .mat file contains a struct with the following keys/matrices:
          - kspace: shape (Nk, Nc, Ny, Nx)
          - smaps: shape (Nc, Ny, Nx)
          - sampling_pattern (optional): shape (Nk, Ny)?
        """
        with h5py.File(matfile_path, 'r', rdcc_nbytes=1024**3, rdcc_w0=1, rdcc_nslots=1024) as f:

            sampling_pattern = None
            if "sampling_pattern" in f.keys():
                sampling_pattern = torch.tensor(np.array(f["sampling_pattern"]).squeeze())

            smaps = to_tensor(h5py2Complex(f['smaps']))
            kspace_sense = to_tensor(h5py2Complex(f['kspace'], load_in_chunks=load_in_chunks)) # replace this line and avoid duplicate data to save memory

        assert smaps.ndim == 4 and kspace_sense.ndim == 5

        (Nc, Ny, Nx, _) = smaps.shape
        (Nk, _, Ny_sense, Nx_sense, _) =  kspace_sense.shape

        assert Nx == Nx_sense, "kx-space is assumed to have correct dimension"

        if Ny != Ny_sense: # non-zero kspace data is stored packed together -> needs to be spaced evenly
            
            K_sense = math.floor(Ny / Ny_sense) # SENSE fator

            # zero-interleave the sampled k-space data -> the phase axis is sampled with uniform spacing
            kspace = torch.zeros((Nk, Nc, Ny, Nx, 2), dtype=torch.float32)
            kspace[:,:,2:(K_sense*Ny_sense+2):K_sense,:] = kspace_sense

            # create the sampling mask that corresponds to the zero-interleaving
            mask = ((torch.sum(torch.abs(kspace), dim=1) > 0)*1.).type(np.float32)
        
        else: # kspace data already has the correct shape (same as smaps)
            kspace = kspace_sense

            # determine the mask by identifying non-zero entries in the k-space matrix # ! might be incorrect in some instances if all coils truely measure zero exactely
            mask = ((torch.sum(torch.abs(kspace), dim=1) > 0)*1.).type(np.float32)
            if sampling_pattern is not None:
                mask *= torch.tile(sampling_pattern.reshape((Ny, 1)), (1, Nx))
                for k in range(Nk):
                    for c in range(Nc):
                        kspace[k,c,:,:] *= mask[k,:,:]

        kspace = kspace.unsqueeze(dim=2)
        mask = mask.unsqueeze(dim=1)
        smaps = smaps.unsqueeze(dim=1)

        return self(kspace, mask, smaps, transform=transform)

    @classmethod
    def from_npyfiles(self, base_path_and_name, transform=None):
        """
        Loads a dataset from .npy files.

        Following files are loaded:
          - `<base_path_and_name>_kspace.npy`
          - `<base_path_and_name>_mask.npy`
          - `<base_path_and_name>_smaps.npy`
          - `<base_path_and_name>_add.pth` (if additional_data exists)
          - `<base_path_and_name>_ref.npy` (if reference exists)

        Use `dataset.saveToNpyFiles` for generating the mentioned .npy files from other dataset formats.
        """

        kspace = np.load("{}_kspace.npy".format(base_path_and_name))
        mask = np.load("{}_mask.npy".format(base_path_and_name))
        smaps = np.load("{}_smaps.npy".format(base_path_and_name))

        reference = None
        if os.path.isfile("{}_ref.npy".format(base_path_and_name)):
            reference = np.load("{}_ref.npy".format(base_path_and_name))

        additional_data = None
        if os.path.isfile("{}_add.pth".format(base_path_and_name)):
            additional_data = torch.load("{}_add.pth".format(base_path_and_name))
            
        line_indices = None
        if os.path.isfile("{}_lines.npy".format(base_path_and_name)):
            line_indices = np.load("{}_lines.npy".format(base_path_and_name))


        if kspace.ndim == 4: # this is 2D dataset -> expand to 3D
            kspace = np.expand_dims(kspace, axis=2)
            mask = np.expand_dims(mask, axis=1)
            smaps = np.expand_dims(smaps, axis=1)
            if reference is not None: reference = np.expand_dims(reference, axis=1)

        # convert to torch.tensor
        kspace = to_tensor(kspace)
        mask = torch.tensor(mask, dtype=torch.float32)
        smaps = to_tensor(smaps)
        if reference is not None: reference = torch.tensor(reference)

        return self(kspace, mask, smaps, reference, transform, additional_data=additional_data, line_indices=line_indices)


    @classmethod
    def from_sparse_matfile2d(self, matfile_path, listfile_path, transform=None, shift=False, skip_outliers=False, remove_padding=False, set_smaps_outside_to_one=False):
        """ 
        Loads `matfile_path` that stores measurement data without zeros in the k-space (sparse respresentation of data)
        -> the measured ky-lines are copied to the correct positions in the k-space matrix by this method

        Requires the `matfile_path + '.list'` file for copying the measurements to the correct location in the k-space matrix.

        Parameters:
        - `transform`: An optional function that is applied to every sample data loaded from the dataset with the get_item() method.
        - `shift`: If true, the image is shifted by Ny/4. This is necessary if the k-space data does not match the smaps otherwise.
        - `skip_outliers`: If true, the method does not raise an exception if measured ky-coordinates are outside of the range of the k-space matrix.
        - `remove_padding`: By default, the k-space data of the scanner is zero-padded and the smaps are computed on a larger grid. If `remove_padding` is true, the padding is removed and the smaps are cropped in the Fourier domain.
        - `set_smaps_outside_to_one`: By default, the smaps estimated by the scanner have zero entries outside the human body (they cannot be estimated outside, as there is no signal). Thus, the reconstructions can take arbitrary values outside the body without affecting the reconstruction loss. If `set_smaps_outside_to_one` is true, the zero-sensitivities are set to 1.0. By setting the smaps outside to 1.0, the reconstructions are forced to zero outside the human body.
        """

        list_data = ListData(file_name=listfile_path)

        # load matrices from the .mat file
        self.noise = None
        with h5py.File(matfile_path, 'r', rdcc_nbytes=1024**3, rdcc_w0=1, rdcc_nslots=1024) as f:
            raw_smaps = h5py2Complex(f["smaps"], load_in_chunks=False)
            raw_kspace = h5py2Complex(f["kspace"], load_in_chunks=False)
            reference = np.array(f["reference"])
            if "noise" in f.keys():
                # contains noise measurements that can be used to estimate noise statistics of the receiver coils
                self.noise = torch.tensor(h5py2Complex(f["noise"], load_in_chunks=False))
            encoding_pars = parse_struct(f["encoding_pars"])


        # detect datasets that are binned by cardiac phases -> handle them separately as they probably use the SENSE pattern
        if list_data.Nk_card > 1:
            Nk = list_data.Nk_card            
        else:
            Nk = list_data.Nk

        # if required, set the zero pixels of the smaps (outside the body) to 1.0
        if set_smaps_outside_to_one:
            raw_smaps[raw_smaps == 0.] = 1.
        smaps = to_tensor(raw_smaps)

        Nc = smaps.shape[0]
        # if required, remove the zero-padding from the k-space and truncate the smaps in the Fourier domain
        k_sense = 1
        if remove_padding:
            # compute the resolution in x-direction without padding
            Nx = int(encoding_pars["KxRange"][1] - encoding_pars["KxRange"][0] + 1)

            if list_data.Nk_card > 1:
                # compute the SENSE factor
                k_sense = int(encoding_pars["YRes"] / (int(encoding_pars["KyRange"][1] - encoding_pars["KyRange"][0] + 1)))
                Ny = int(encoding_pars["KyRange"][1] - encoding_pars["KyRange"][0] + 1) * k_sense
            else:
                # compute the resolution in y-direction without padding
                if -encoding_pars["KyRange"][0] == encoding_pars["KyRange"][1] + 1: # standard case
                    Ny = int(encoding_pars["KyRange"][1] - encoding_pars["KyRange"][0] + 1)
                elif -encoding_pars["KyRange"][0] < encoding_pars["KyRange"][1]: # probably partial-Fourier
                    Ny = int(2 * encoding_pars["KyRange"][1])
                else: # unknown case
                    print(encoding_pars["KyRange"], smaps.shape[2])
                    raise Exception

            _, Nyold, Nxold, _ = smaps.shape

            # truncate the smaps in the Fourier domain
            Nystart, Nxstart = int((Nyold - Ny) / 2), int((Nxold - Nx) / 2)

            smaps_fft = fft2(smaps)
            smaps_fft = smaps_fft[:,Nystart:(Nystart+Ny), Nxstart:(Nxstart+Nx), :]
            smaps = ifft2(smaps_fft)

        else:
            Nc, Ny, Nx, _ = smaps.shape
        
        # create zero matrices where the sparse data is filled into
        kspace = torch.zeros((Nk, Nc, Ny, Nx, 2), dtype=torch.float32)
        mask = torch.zeros((Nk, Ny, Nx), dtype=torch.float32)

        num_lines, Nx_raw = raw_kspace.shape

        if list_data.Nk_card > 1:
            # get a lists that contains the following information for every ky-line in the matrix `raw_kspace`: index of the cardiac dynamic, index of the coil, ky indices (shifted)
            dynamics, coil_indices, ky_indices = list_data.get_cardiac_bins_channel_indices_and_kyindices()
        else:
            # get a lists that contains the following information for every ky-line in the matrix `raw_kspace`: index of the dynamic, index of the coil, ky indices (shifted)
            dynamics, coil_indices, ky_indices = list_data.get_dynamics_channel_indices_and_kyindices()
        

        # find the zero index of the k-space matrix in ky-direction
        ky_zero_index = int(Ny / 2)

        # find the first kx index that should be filled with data
        kx_shift = int(Nx / 2 - Nx_raw / 2)

        for i in range(num_lines):
            k = dynamics[i]
            c = coil_indices[i]
            ky = ky_indices[i]

            ky_shifted = ky_zero_index + ky * k_sense

            if skip_outliers and ky > Ny / 2 or ky < -Ny / 2:
                continue

            mask[k, ky_shifted, kx_shift:kx_shift+Nx_raw] = 1.
            if shift:
                kspace[k, c, ky_shifted, kx_shift:kx_shift+Nx_raw, :] = to_tensor(raw_kspace[i, :] * np.exp(1j*np.pi*(ky_shifted - Ny/2)))
            else:
                kspace[k, c, ky_shifted, kx_shift:kx_shift+Nx_raw, :] = to_tensor(raw_kspace[i, :])
            
        # if reference data is available, the reference matrix has at least 3 dimensions
        if reference.ndim < 3:
            reference = None

        # insert dimensions for the z-axis (that is not used since this method handles 2D datasets with a single slice)
        kspace = kspace.unsqueeze(dim=2)
        mask = mask.unsqueeze(dim=1)
        smaps = smaps.unsqueeze(dim=1)
        if reference is not None: reference = torch.tensor(reference).unsqueeze(dim=1)

        return self(kspace, mask, smaps, reference=reference, transform=transform)


    @classmethod
    def rebin_cartesian_dataset_extract_validationset(self, dataset, listfile_path, number_of_lines_per_frame, validation_percentage, seed=1998, max_Nk=-1, transform=None, validation_transform=None):
        """
        This method extracts a validation dataset from the sampled lines and rebins the remaining lines into frames with `number_of_lines_per_frame`.

        Parameters:
        - `dataset`: the CartesianDataset that needs to be rebinned
        - `listfile_path`: path to the .list file that belongs to the dataset (used for looking-up the order of the sampled lines)
        - `number_of_lines_per_frame`: number of lines per frame after re-binning
        - `validation_percentage`:  percentage of the sampled ky-lines that is randomly extracted as a validation set
        - `sample_period`: sample period between consecutive ky-lines (=TR for single-echo sequences). This information is used to compute the frame times t_k of the rebinned frames that are stored in the `additional_data` attribute of the retured rebinned dataset.
        - `seed`: seed for the random number generator that randomly selects the validation lines
        - `max_Nk`: number of frames in the rebinned dataset. If not all all frames are needed for training, setting `max_Nk` to the required number of frames saves RAM.

        Outputs: (dataset, validation_set)
        - The `dataset` is a shares the same class as the input dataset.
        - The `validation_set` contains the measuremnts and sampling masks of the validation lines.
        """
        
        random_state = np.random.get_state() # save the current state of the RNG for restoring it later
        np.random.seed(seed) # set the specified random seed
        
        # load the .list file that contains information about the order in which lines were sampled
        listdata = ListData(listfile_path)
        dynamics, ky_indices = listdata.get_dynamics_and_kyindices()
        N_ky_lines = len(ky_indices)

        # generate a random validation subset:
        validation_subset = np.arange(stop=N_ky_lines)
        np.random.shuffle(validation_subset)
        validation_subset = validation_subset[0:int(N_ky_lines * validation_percentage / 100)]

        # re-bin lines
        Nk = (N_ky_lines - len(validation_subset)) // number_of_lines_per_frame
        if max_Nk > 0:
            Nk = min(Nk, max_Nk)
        _, Nc, _, Ny, Nx = dataset.shape()

        kspace = torch.zeros((Nk, Nc, 1, Ny, Nx, 2), dtype=torch.float32)
        mask = torch.zeros((Nk, 1, Ny, Nx), dtype=torch.float32)
        line_indices = torch.zeros((Nk, number_of_lines_per_frame), dtype=torch.int64)

        # count the number of validation lines within Nk
        j = 0
        Nk_validation = 0
        for k in range(Nk):
            for l in range(number_of_lines_per_frame):
                while j in validation_subset:
                    Nk_validation += 1
                    j += 1
                j += 1

        ky_zero_index = int(Ny / 2)

        # SparseCartesianDataset tensors of the validation dataset
        Nr = torch.nonzero(dataset.mask[dynamics[0], 0, ky_zero_index + ky_indices[0], :])[:, 0].shape[0]
        kspace_validation = torch.zeros((Nk_validation, Nc, 1, Nr, 2), dtype=torch.float32)
        mask_validation = torch.zeros((Nk_validation, 1, Nr, 3), dtype=torch.int64)
        trajectory_validation = torch.zeros((Nk_validation, 1, Nr, 3), dtype=torch.float32)
        line_indices_validation = torch.zeros((Nk_validation, 1), dtype=torch.int64)

        j = 0
        k_validation = 0
        for k in range(Nk):
            for i in range(number_of_lines_per_frame):
                while j in validation_subset: # extract the line into the validation set
                    ky = ky_zero_index + ky_indices[j] # get ky index in kspace tensor

                    # convert mask and kspace to sparse representation
                    kx_indices = torch.nonzero(dataset.mask[dynamics[j], 0, ky, :])[:, 0]
                    mask_sparse = torch.stack((torch.zeros(Nr), torch.ones(Nr) * ky, kx_indices), axis=1).reshape((Nr,3)).type(torch.int64)
                    kspace_sparse = dataset.kspace[dynamics[j], :, 0, ky, kx_indices, :].reshape((Nc, Nr, 2))
                    trajectory = mask_to_trajectory(mask_sparse, 1, Ny, Nx)

                    kspace_validation[k_validation, :, 0, :, :] = kspace_sparse
                    trajectory_validation[k_validation, 0, :, :] = trajectory
                    mask_validation[k_validation, 0, :, :] = mask_sparse
                    line_indices_validation[k_validation, 0] = j

                    k_validation += 1
                    j += 1
                
                ky = ky_zero_index + ky_indices[j] # get ky index in kspace tensor
                kspace[k, :, :, ky, :] = dataset.kspace[dynamics[j], :, :, ky, :]
                mask[k, :, ky, :] = dataset.mask[dynamics[j], :, ky, :]
                line_indices[k, i] = j

                j += 1

        reference = dataset.reference

        np.random.set_state(random_state) # restore the state of the RNG
        return self(kspace, mask, dataset.smaps, line_indices=line_indices, reference=reference, transform=transform), SparseCartesianDataset(kspace_validation, trajectory_validation, mask_validation, line_indices_validation, dataset.smaps, transform=validation_transform)

    @classmethod
    def from_cartesian_dataset(self, dataset, transform=None):
        """
        Creates a shallow copy (underlying data remains identical).
        
        Use this method to initialize many different subclasses from the same data,
        without loading the data into memory multiple times. Example usage:
        ```python
        dataset = CartesianDataset.fromNpyFiles(...)
        slice_dataset = CartesianSliceDataset.from_cartesian_dataset(dataset, transform=...)
        ```
        """
        return self(dataset.kspace, dataset.mask, dataset.smaps, reference=dataset.reference, transform=transform, additional_data=dataset.additional_data, line_indices=dataset.line_indices)
    

    def shape(self):
        """
        Returns (Nk, Nc, Ny, Nx).
        """
        return (self.Nk, self.Nc, self.Nz, self.Ny, self.Nx)

    
    def saveToNpyFiles(self, base_path_and_name):
        """
        Saves the entire dataset to .npy files that can be loaded efficiently.
        """

        def to_numpy(tensor):
            if tensor.shape[-1] == 2: # interpret as complex
                return np.array(tensor[..., 0] + 1j*tensor[..., 1], dtype=np.complex64)
            return np.array(tensor)

        with open("{}_kspace.npy".format(base_path_and_name), "wb") as f:
            np.save(f, to_numpy(self.kspace))
        with open("{}_smaps.npy".format(base_path_and_name), "wb") as f:
            np.save(f, to_numpy(self.smaps))
        with open("{}_mask.npy".format(base_path_and_name), "wb") as f:
            np.save(f, to_numpy(self.mask))
        if not self.reference is None:
            with open("{}_ref.npy".format(base_path_and_name), "wb") as f:
                np.save(f, to_numpy(self.reference))
        if self.additional_data is not None:
            torch.save("{}_add.pth".format(base_path_and_name), self.additional_data)


    def subset(self, subset_indices):
        return Subset(self, subset_indices)


class CartesianSliceDataset(CartesianDataset):
    """
    Returns frames of a selected slice.
    """

    def __len__(self):
        return self.Nk

    def __getitem__(self, kz):
        """
            Valid formats with z=0:
             - dataset[k]
             - dataset[[0,1,2,3]]
             - dataset[1:5]

            Valid formats with selected slice z:
             - dataset[k,z]
             - dataset[[0,1,2,3], z]
             - dataset[1:5, z]
        """
        
        if isinstance(kz, tuple):
            k, z = kz
            assert isinstance(z, int)
        else:
            k = kz
            z = 0

        if isinstance(k, int):
            indices = [k]
        elif isinstance(k, slice):
            indices = range(k.start or 0, k.stop or self.Nk, k.step or 1)
        elif isinstance(k, Iterable):
            indices = k
        else:
            raise Exception("invalid index format")

        Nk = len(indices)
        mask = torch.zeros((Nk, self.Ny, self.Nx), dtype=torch.float32)
        kspace = torch.zeros((Nk, self.Nc, self.Ny, self.Nx, 2), dtype=torch.float32)
        for n, index in enumerate(indices):
            mask[n, :, :] = self.mask[index, z, :, :]
            kspace[n, :, :, :, :] = self.kspace[index, :, z, :, :, :]

        reference = None
        if self.reference is not None:
            reference = torch.zeros((Nk, self.Ny, self.Nx), dtype=torch.complex64)
            for n, index in enumerate(indices):
                reference[n, :, :] = self.reference[index, z, :, :]

        line_indices = None
        if self.line_indices is not None:
            line_indices = self.line_indices[indices, :]

        sample = {
            'indices': torch.tensor(indices, dtype=torch.int64),
            'z': z,
            'kspace': kspace,
            'smaps': self.smaps,
            'mask': mask,
            'reference': reference,
            'shape': (Nk, self.Nc, 1, self.Ny, self.Nx),
            'additional_data': self.additional_data,
            'line_indices': line_indices
        }

        if(self.transform):
            return self.transform(sample)
        return sample


class SparseCartesianDataset():
    """ 
    This class stores a Cartesian dataset without zeros in the k-space and stores a tensor of measured k-space coordinates (trajectory) instead of a sampling mask. 
    
    attributes:
    - `kspace`: shape (Nl, Nc, Nr, 2) contains the complex k-space measurements (real, imaginary part in last dimension)
    - `trajectory`: shape (Nl, Nr, 3) contains the k-space coordinates of the measurements normalized to 1/FOV. Order of the dimensions: z y x.
    - `mask`: shape (Nl, Nr, 3) contains the indices of the measured coordinates on the Cartesian grid (same information as in trajectory, but in a more accessible format). Order of the dimensions: z y x.
    - `smaps`:  shape (Nc, Nz, Ny, Nx, 2) torch.tensor float32
    - `transform`: optional function: dict -> dict
    - `additional_data`: arbitrary data

    naming:
    - `Nk`: number of frames
    - `Nl`: number of measured k-space lines
    - `Nr`: number if measurements in the read-out direction (x-direction)
    - `Nc`: number of receiver coils
    - `Nz`, `Ny`, `Nx`: resolution in z, y, and x-direction
    """

    def __init__(self, kspace, trajectory, mask, line_indices, smaps, reference=None, transform=None, additional_data=None):
        
        self.kspace = kspace # shape (Nk, Nc, Nl, Nr, 2) torch.tensor float32
        self.trajectory = trajectory # shape (Nk, Nl, Nr, 3) torch.tensor float32
        self.mask = mask # shape (Nk, Nl, Nr, 3) LongTensor
        self.line_indices = line_indices # shape (Nk, Nl) LongTensor
        self.smaps = smaps # shape (Nc, Nz, Ny, Nx, 2) torch.tensor float32
        self.transform = transform # function: transform(dict: sample) -> dict
        self.reference = reference
        self.additional_data = additional_data # arbitrary data 

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
    def from_sparse_matfile2d_extract_validation_dataset_rebin(self, matfile_path, listfile_path,
                                                         transform=None,
                                                         shift=False,
                                                         remove_padding=False,
                                                         set_smaps_outside_to_one=False,
                                                         validation_percentage=0.,
                                                         number_of_lines_per_frame=6,
                                                         max_Nk=-1,
                                                         seed=1998
                                                         ):
        """ 
        Loads `matfile_path` that stores measurement data without zeros in the k-space (sparse respresentation of data).
        Requires `listfile_path` that contains information about the order and position of the measured k-space lines.
        Randomly extracts `validation_percentage` percent of the k-space lines for validation.
        Bins the remaining lines into frames with `number_of_lines_per_frame` lines each. If `max_Nk` is specified, the number of frames is reduced and excess frames are discarded.


        Arguments:
        - `transform`: An optional function that is applied to every sample data loaded from the dataset with the get_item() method.
        - `shift`: If true, the image is shifted by Ny/4. This is necessary if the k-space data does not match the smaps otherwise.
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

        list_data = ListData(file_name=listfile_path)

        # detect datasets that are binned by cardiac phases -> handle them separately as they probably use the SENSE pattern
        assert list_data.Nk_card == 1, "ECG-binned data needs to be loaded differently (Nk_card > 1, Nk == 1). Not implemented yet for sparse datasets."
        
        # load matrices from the .mat file
        with h5py.File(matfile_path, 'r', rdcc_nbytes=1024**3, rdcc_w0=1, rdcc_nslots=1024) as f:
            raw_smaps = h5py2Complex(f["smaps"], load_in_chunks=False)
            raw_kspace = h5py2Complex(f["kspace"], load_in_chunks=False)
            reference = np.array(f["reference"])
            encoding_pars = parse_struct(f["encoding_pars"])
            self.encoding_pars = encoding_pars

        # if required, set the zero pixels of the smaps (outside the body) to 1.0
        if set_smaps_outside_to_one:
            raw_smaps[raw_smaps == 0.] = 1.

        # if required, remove the zero-padding from the k-space and truncate the smaps in the Fourier domain
        if remove_padding: # remove the padding

            # compute the resolution in x-direction without padding
            Nx = int(encoding_pars["KxRange"][1] - encoding_pars["KxRange"][0] + 1)
            Nc, Nyold, Nxold = raw_smaps.shape

            # compute the resolution in y-direction without padding
            if -encoding_pars["KyRange"][0] == encoding_pars["KyRange"][1] + 1: # standard case
                Ny = int(encoding_pars["KyRange"][1] - encoding_pars["KyRange"][0] + 1)
            elif -encoding_pars["KyRange"][0] < encoding_pars["KyRange"][1]: # probably partial-Fourier
                Ny = int(2 * encoding_pars["KyRange"][1])
            else: # unknown case
                print(encoding_pars["KyRange"], raw_smaps.shape[2])
                raise Exception

            # truncate the smaps in the Fourier domain
            Nystart, Nxstart = int((Nyold - Ny) / 2), int((Nxold - Nx) / 2)

            smaps = to_tensor(raw_smaps)
            smaps_fft = fft2(smaps)
            smaps_fft = smaps_fft[:,Nystart:(Nystart+Ny), Nxstart:(Nxstart+Nx), :]
            smaps = ifft2(smaps_fft)

        else: # keep the padding
            Nc, Ny, Nx = raw_smaps.shape
            smaps = to_tensor(raw_smaps)
            
        # get a lists that contains the following information for every ky-line in the matrix `kspace`: index of the dynamic, index of the coil, ky indices (shifted)
        dynamics, coil_indices, ky_indices = list_data.get_dynamics_channel_indices_and_kyindices()
        num_lines_all_coils, Nr = raw_kspace.shape
        assert num_lines_all_coils % Nc == 0
        Nl = num_lines_all_coils // Nc
        
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
        kspace = np.zeros((Nk, Nc, number_of_lines_per_frame, Nr), dtype=np.csingle)
        mask = np.zeros((Nk, number_of_lines_per_frame, Nr, 3), dtype=np.int64)
        trajectory = np.zeros((Nk, number_of_lines_per_frame, Nr, 3), dtype=np.float32)
        line_indices = np.zeros((Nk, number_of_lines_per_frame), dtype=np.int64)

        kspace_validation = np.zeros((Nk_validation, Nc, 1, Nr), dtype=np.csingle)
        mask_validation = np.zeros((Nk_validation, 1, Nr, 3), dtype=np.int64)
        trajectory_validation = np.zeros((Nk_validation, 1, Nr, 3), dtype=np.float32)
        line_indices_validation = np.zeros((Nk_validation, 1), dtype=np.int64)

        # find the zero index of the k-space matrix in ky-direction
        ky_zero_index = int(Ny / 2)
        # find the first kx index that should be filled with data
        kx_shift = int(Nx / 2 - Nr / 2)

        # stacks the measurements from different coils
        def line_generator():
            for l in range(Nl):
                line_kspace = np.zeros((Nc, Nr), dtype=np.csingle)
                for c in range(Nc):
                    i = l*Nc+c
                    assert dynamics[i] == dynamics[l*Nc]
                    assert ky_indices[i] == ky_indices[l*Nc]
                    line_kspace[coil_indices[i], :] = raw_kspace[i, :]

                ky = ky_indices[l*Nc]
                ky_index = ky_zero_index + ky
                ky_coordinate = 2. * np.pi * ky / Ny

                if shift:
                    line_kspace *= np.exp(np.pi*1j*(ky_index - Ny/2))

                kx_indices = kx_shift + np.arange(Nr)
                kx_coordinates = 2. * np.pi * (-int(Nr/2) + np.arange(Nr)) / Nx

                line_mask = np.stack((np.zeros(Nr), np.ones(Nr)*ky_index, kx_indices), axis=-1)
                line_trajectory = np.stack((np.zeros(Nr), np.ones(Nr)*ky_coordinate, kx_coordinates), axis=-1)

                yield line_kspace, line_trajectory, line_mask


        lines = line_generator()
        j = 0
        k_validation = 0
        for k in range(Nk):
            for l in range(number_of_lines_per_frame):
                while j in validation_indices: # put lines in the validation dataset
                    line_kspace, line_trajectory, line_mask = next(lines)

                    kspace_validation[k_validation, :, 0, :] = line_kspace
                    trajectory_validation[k_validation, 0, :, :] = line_trajectory
                    mask_validation[k_validation, 0, :, :] = line_mask
                    line_indices_validation[k_validation, 0] = j

                    k_validation += 1
                    j += 1
                
                # put the line in the training dataset
                line_kspace, line_trajectory, line_mask = next(lines)
                kspace[k, :, l, :] = line_kspace
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
        
        
        # if reference data is available, the reference matrix has at least 3 dimensions
        if reference.ndim < 3:
            reference = None
        if reference is not None: reference = to_tensor(np.expand_dims(reference, axis=1))

        # restore the state of the RNG
        np.random.set_state(random_state)
        return self(kspace, trajectory, mask, line_indices, smaps, reference=reference, transform=transform), self(kspace_validation, trajectory_validation, mask_validation, line_indices_validation, smaps, transform=transform)
    
    @classmethod
    def from_cartesian_dataset_extract_validation_dataset_rebin(self, cartesian_dataset, listfile_path,
                                                         transform=None,
                                                         validation_percentage=0.,
                                                         number_of_lines_per_frame=6,
                                                         max_Nk=-1,
                                                         seed=1998):
        # save the current state of the RNG and set the new one
        random_state = np.random.get_state() 
        np.random.seed(seed) 

        list_data = ListData(file_name=listfile_path)

        # get a lists that contains the following information for every ky-line in the matrix `kspace`: index of the dynamic, index of the coil, ky indices (shifted)
        dynamics, ky_indices = list_data.get_dynamics_and_kyindices()
        Nl = len(ky_indices)
        Nr = cartesian_dataset.Nx # sample the entire ky line (!! cartesian dataset must not have padding)
        
        Nc, Ny, Nx = cartesian_dataset.Nc, cartesian_dataset.Ny, cartesian_dataset.Nx

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
        kspace = torch.zeros((Nk, Nc, number_of_lines_per_frame, Nr, 2), dtype=torch.float32)
        mask = torch.zeros((Nk, number_of_lines_per_frame, Nr, 3), dtype=torch.int64)
        trajectory = torch.zeros((Nk, number_of_lines_per_frame, Nr, 3), dtype=torch.float32)
        line_indices = torch.zeros((Nk, number_of_lines_per_frame), dtype=torch.int64)

        kspace_validation = torch.zeros((Nk_validation, Nc, 1, Nr, 2), dtype=torch.float32)
        mask_validation = torch.zeros((Nk_validation, 1, Nr, 3), dtype=torch.int64)
        trajectory_validation = torch.zeros((Nk_validation, 1, Nr, 3), dtype=torch.float32)
        line_indices_validation = torch.zeros((Nk_validation, 1), dtype=torch.int64)

        # find the zero index of the k-space matrix in ky-direction
        ky_zero_index = int(Ny / 2)
        # find the first kx index that should be filled with data
        kx_shift = 0

        # stacks the measurements from different coils
        def line_generator():
            for i, (k, ky) in enumerate(zip(dynamics, ky_indices)):
                
                ky_index = ky_zero_index + ky
                ky_coordinate = 2. * torch.pi * ky / Ny

                line_kspace = cartesian_dataset.kspace[k, :, 0, ky_index, :, :]

                kx_indices = kx_shift + torch.arange(Nr)
                kx_coordinates = 2. * torch.pi * (-int(Nr/2) + torch.arange(Nr)) / Nx

                line_mask = torch.stack((torch.zeros(Nr), torch.ones(Nr)*ky_index, kx_indices), axis=-1)
                line_trajectory = torch.stack((torch.zeros(Nr), torch.ones(Nr)*ky_coordinate, kx_coordinates), axis=-1)

                yield line_kspace, line_trajectory, line_mask


        lines = line_generator()

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

        reference = cartesian_dataset.reference
        if cartesian_dataset.reference is not None and cartesian_dataset.reference.shape[0] == Nl:
            # phantom dataset -> extract a reference images for each frame
            frame_indices = torch.round(0.5*(line_indices[:,0] + line_indices[:,-1])).type(torch.long)
            reference = cartesian_dataset.reference[frame_indices, ...]

        # restore the state of the RNG
        np.random.set_state(random_state)
        return self(kspace, trajectory, mask, line_indices, cartesian_dataset.smaps, reference=reference, transform=transform), self(kspace_validation, trajectory_validation, mask_validation, line_indices_validation, cartesian_dataset.smaps, transform=transform)
    

        
    @classmethod
    def from_sparse_cartesian_dataset(self, dataset, transform=None):
        """
        Creates a shallow copy (underlying data remains identical).
        
        Use this method to initialize many different subclasses from the same data, without loading the data into memory multiple times.
        """
        return self(dataset.kspace, dataset.trajectory, dataset.mask, dataset.line_indices, dataset.smaps, reference=dataset.reference, additional_data=dataset.additional_data, transform=transform)

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

        reference = None
        if not self.reference is None:
            reference = torch.zeros((len(indices), 1, self.Ny, self.Nx), dtype=torch.complex64)
            for n, index in enumerate(indices):
                reference[n, 0, :, :] = self.reference[index, :, :, :]

        sample = {
            'indices': torch.tensor(indices, dtype=torch.int64),
            'kspace': self.kspace[indices, :, :, :, :],
            'trajectory': self.trajectory[indices, :, :, :],
            'mask': self.mask[indices, :, :, :],
            'line_indices': self.line_indices[indices, :],
            'smaps': self.smaps,
            'reference': reference,
        }

        if(self.transform):
            return self.transform(sample)
        return sample

class ReferenceDataset():
    """ 
    Class for loading only the reference reconstructions from the scanner or any dataset with available reference images.
    """
    def __init__(self, reference):
        self.reference = reference
        self.Nk, self.Nz, self.Ny, self.Nx = reference.shape

    @classmethod
    def from_cartesian_dataset(self, dataset):
        """
        Extract the reference reconstructions from a `CartesianDataset`. This is useful to avoid unnecessary overhead when accessign the references (kspace data is not loaded).
        """
        return self(dataset.reference)

    @classmethod
    def from_npy_files(self, base_path_and_name):
        """
        Load the reference from a `.npy` file of the format (Nk, Ny, Nx) or (Nk, Nz, Ny, Nx)
        """
        reference = np.load("{}_ref.npy".format(base_path_and_name))
        if reference.ndims == 3:
            reference = np.expand_dims(reference, axis=1)
        return self(reference)

    @classmethod
    def from_mat_file_2d(self, matfile_path):
        """
        Load the reference from a `.mat` file.
        """
        with h5py.File(matfile_path, 'r', rdcc_nbytes=1024**3, rdcc_w0=1, rdcc_nslots=1024) as f:
            reference = np.array(f["reference"])
        assert reference.ndim == 3, "There is no reference reconstruction for this dataset."
        reference = np.expand_dims(reference, axis=1)
        return self(reference)

    @classmethod
    def from_dicom(self, dicom_path):
        ds = pydicom.dcmread(dicom_path)
        self.ds = ds
        reference = np.array(ds.pixel_array)
        if reference.ndim == 3: # k y x -> k z y x
            reference = np.expand_dims(reference, axis=1)
        assert reference.ndim == 4
        return self(reference)

    def subset(self, subset_indices):
        """
        Access only a subset of the reference images.
        """
        return Subset(self, subset_indices)
    
    def __len__(self):
        return self.Nk
    
    def __getitem__(self, kz):
        """
            Valid formats with z=0:
             - dataset[k]
             - dataset[[0,1,2,3]]
             - dataset[1:5]

            Valid formats with selected slice z:
             - dataset[k,z]
             - dataset[[0,1,2,3], z]
             - dataset[1:5, z]
        """
        if isinstance(kz, tuple):
            k, z = kz
            assert isinstance(z, int)
        else:
            k = kz
            z = 0

        if isinstance(k, int):
            indices = [k]
        elif isinstance(k, slice):
            indices = range(k.start or 0, k.stop or self.Nk, k.step or 1)
        elif isinstance(k, Iterable):
            indices = k
        else:
            raise Exception("invalid index format")

        Nk = len(indices)
        
        reference = np.zeros((Nk, 1, self.Ny, self.Nx), dtype=np.csingle)
        for n, index in enumerate(indices):
            reference[n, 0, :, :] = self.reference[index, z, :, :]
        return reference


class NonCartesianDataset2D():
    """ 
    Dataset for non-cartesian 2D multi-slice scans.
    """
    def __init__(self, kspace, traj, smaps, reference, transform):
        pass

class NonCartesianDataset3D():
    """ 
    Dataset for non-cartesian 3D scans.
    """
    def __init__(self, kspace, trajectory, shape, line_indices=None, smaps=None, weights=None, transform=None, reference=None, additional_data=None):
        self.kspace = kspace # (Nk, Nc, Nl, L, 2) float32
        self.trajectory = trajectory # (Nl, L, 3) float32, order of coordinates: z, y, x
        self.smaps = smaps # (Nc, Nz, Ny, Nx, 2) float32
        self.weights = weights # (Nl, L) complex64,
        self.line_indices = line_indices
        self.transform = transform
        self.reference = reference
        self.additional_data = additional_data
        self.Nk, self.Nc, self.Nz, self.Ny, self.Nx = shape
    
    @classmethod
    def from_mat_file_binned_validation(self, matfile_path, listfile_path, number_of_lines_per_frame, validation_percentage, transform=None, load_in_chunks=False, seed=1998, convert_to_rad=True, has_norm_fac=False, transpose_smaps=False, max_Nk=-1):
        """
        Loads a non-cartesian dataset from a Matlab file, extracts a validation dataset, and bins the remaining lines into frames.

        If the following error occurs, `has_norm_fac` needs to be set to True:
            max() arg is an empty sequence
        """
        random_state = np.random.get_state() # only change the seed locally -> restore later
        np.random.seed(seed)
        with h5py.File(matfile_path, 'r', rdcc_nbytes=1024**3, rdcc_w0=1, rdcc_nslots=1024) as f:
            smaps = to_tensor(h5py2Complex(f['smaps']))
            kspace = h5py2Complex(f['kspace'], load_in_chunks=load_in_chunks) # replace this line and avoid duplicate data to save memory
            trajectory = torch.tensor(f['traj']['Kpos'])[:, 0, :, :]
            weights = torch.tensor(np.array(f['traj']['Weights']))
            [Nx_traj], [Ny_traj] = f['traj']['OutputMatrixSize']
            assert Nx_traj == Ny_traj # order of Nx and Ny might be wrong. Not tested yet -> raise Exception.

        # flip order of the coordinates to z, y, x
        trajectory = torch.flip(trajectory, dims=(0,))

        # exemplary kspace shape     (4, 25, 1056, 264)
        # exemplary trajectory shape (3, 1056, 264)
            
        if convert_to_rad:
            trajectory[2, :, :] = trajectory[2, :, :] * (2. * np.pi / Nx_traj)
            trajectory[1, :, :] = trajectory[1, :, :] * (2. * np.pi / Ny_traj)


        listfile = ListData(listfile_path, has_norm_fac=has_norm_fac)
        dynamics, ky_indices = listfile.get_dynamics_and_kyindices()

        num_dyn, Nc, Nl, readout_len = kspace.shape

        num_lines = len(dynamics)

        validation_subset = np.arange(num_lines, dtype=np.int32)
        np.random.shuffle(validation_subset)
        validation_subset = validation_subset[0:int(num_lines * validation_percentage / 100)]

        num_training_lines = num_lines - len(validation_subset)

        Nk = int(math.floor(num_training_lines / number_of_lines_per_frame))
        if max_Nk > 0:
            Nk = min(Nk, max_Nk)

        # count the number of validation lines within Nk
        j = 0
        Nk_validation = 0
        for k in range(Nk):
            for l in range(number_of_lines_per_frame):
                while j in validation_subset:
                    Nk_validation += 1
                    j += 1
                j += 1
        

        kspace_binned = torch.empty((Nk, Nc, number_of_lines_per_frame, readout_len, 2), dtype=torch.float32)
        trajectory_binned = torch.empty((Nk, number_of_lines_per_frame, readout_len, 3), dtype=torch.float32)
        weights_binned = torch.empty((Nk, number_of_lines_per_frame, readout_len), dtype=torch.float32)
        line_indices = torch.empty((Nk, number_of_lines_per_frame), dtype=torch.int64)

        kspace_validation = torch.empty((Nk_validation, Nc, 1, readout_len, 2), dtype=torch.float32)
        trajectory_validation = torch.empty((Nk_validation, 1, readout_len, 3), dtype=torch.float32)
        weights_validation = torch.empty((Nk_validation, 1, readout_len), dtype=torch.float32)
        line_indices_validation = torch.empty((Nk_validation, 1), dtype=torch.int64)

        j = 0
        k_validation = 0
        for k in range(Nk):
            for i in range(number_of_lines_per_frame):
                while j in validation_subset:
                    ky = ky_indices[j]
            
                    kspace_validation[k_validation, :, 0, :] = to_tensor(kspace[dynamics[j], :, ky, :])
                    trajectory_validation[k_validation, 0, :, :] = trajectory[:, ky, :].T
                    weights_validation[k_validation, 0, :] = weights[ky, :]
                    line_indices_validation[k_validation, 0] = j

                    k_validation += 1
                    j += 1
                
                ky = ky_indices[j]
                kspace_binned[k, :, i, :] = to_tensor(kspace[dynamics[j], :, ky, :])
                trajectory_binned[k, i, :, :] = trajectory[:, ky, :].T
                weights_binned[k, i, :] = weights[ky, :]
                line_indices[k, i] = j

                j += 1        


        if smaps.ndim == 4:
            smaps = smaps.unsqueeze(dim=1)
        else:
            assert False, "Warning: convert_to_rad is not implemented for 3D datasets."

        if transpose_smaps:
            smaps = smaps.permute((0, 1, 3, 2, 4))

        Nc, Nz, Ny, Nx, _ = smaps.shape      
        shape = (Nk, int(Nc), int(Nz), int(Ny), int(Nx))
        shape_validation = (Nk_validation, int(Nc), int(Nz), int(Ny), int(Nx))

        if not (Ny == Ny_traj and Nx == Nx_traj):
            print("Warning: trajectory output matrix size does not match the dimensions of the sensitivity maps.")

        np.random.set_state(random_state)
        return self(kspace_binned, trajectory_binned, shape, smaps=smaps, weights=weights_binned, transform=transform, line_indices=line_indices, reference=None), self(kspace_validation, trajectory_validation, shape_validation, smaps=smaps, weights=weights_validation, transform=transform, line_indices=line_indices_validation, reference=None)
    
    def __len__(self):
        """
        Returns the number of frames.
        """
        return self.Nk
    
    def __getitem__(self, k):
        """
        Returns frames k.
        """

        if isinstance(k, int):
            indices = [k]
        elif isinstance(k, slice):
            indices = range(k.start or 0, k.stop or self.Nk, k.step or 1)
        elif isinstance(k, Iterable):
            indices = k
        else:
            raise Exception("invalid index format")
        
        reference = None
        if self.reference is not None:
            reference = self.reference[indices, :, :]

        line_indices = None
        if self.line_indices is not None:
            line_indices = self.line_indices[indices]

        sample = {
            "indices": torch.tensor(indices, dtype=torch.int64),
            "kspace": self.kspace[indices],
            "smaps": self.smaps,
            "trajectory": self.trajectory[indices],
            "weights": self.weights[indices],
            "shape": (self.Nk, self.Nc, self.Nz, self.Ny, self.Nx),
            "reference": reference,
            "line_indices": line_indices,
            "additional_data": self.additional_data
        }

        if(self.transform):
            return self.transform(sample)
        return sample

    def shape(self):
        """
        Returns (Nk, Nc, Nz, Ny, Nx).
        """
        return (self.Nk, self.Nc, self.Nz, self.Ny, self.Nx)


    def subset(self, subset_indices):
        return Subset(self, subset_indices)
    

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


class DatasetCache(CartesianDataset):
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



class ListData():
    """
    Class for parsing `.list` files containing measurement metadata on samples kspace lines, coils, etc.
    """
    def __init__(self, file_name, has_norm_fac=False):
        
        if has_norm_fac:
            self.key_names = ['typ', 'mix', 'dyn', 'card', 'echo', 'loca', 'chan', 'extr1', 'extr2', 'ky', 'kz', 'n.a.', 'aver', 'sign', 'rf', 'grad', 'enc', 'rtop', 'rr', 'size', 'offset', 'rphase', 'mphase', 'pda', 'pda_f', 't_dyn', 'codsiz', 'chgrp', 'format', 'norm_fac', 'kylab', 'kzlab']
        else:
            self.key_names = ['typ', 'mix', 'dyn', 'card', 'echo', 'loca', 'chan', 'extr1', 'extr2', 'ky', 'kz', 'n.a.', 'aver', 'sign', 'rf', 'grad', 'enc', 'rtop', 'rr', 'size', 'offset', 'rphase', 'mphase', 'pda', 'pda_f', 't_dyn', 'codsiz', 'chgrp', 'format', 'kylab', 'kzlab']
        

        self.data = pd.read_csv(file_name, sep=' ', skiprows=5, header=None, names=self.key_names, dtype={'ID':str}, skipinitialspace=True)

        self.measurements = self.data[self.data["typ"] == 1]
        self.Nk = self.measurements["dyn"].max() + 1
        self.Nk_card = self.measurements["card"].max() + 1
    
    def get_channel_numbers(self):
        """
        Returns the set of channel numbers of active receiver coils.
        """
        return np.array(self.measurements["chan"].unique())

    def get_kyindices(self):
        """
        Returns the set of ky-indices that are sampled in the scan.
        """
        first_channel_number = self.get_channel_numbers()[0]
        return np.array(self.measurements[self.measurements["chan"] == first_channel_number]["ky"])

    def get_sample_time(self):
        """
        Returns the physical time of all measurements.
        """
        assert self.Nk_card == 1, "not implemented otherwise"
        assert len(self.measurements["echo"].unique()) == 1, "not implemented otherwise"
        return np.array(self.measurements["dyn"]) * self.t_r

    def get_dynamics_channel_indices_and_kyindices(self):
        """
        Returns the index k of the dynamic, the channel c (in 0...C-1), and the ky index of all measurements in chronological order.
        """
        channels_numbers = self.get_channel_numbers()
        inverse_map = np.zeros(max(channels_numbers)+1, dtype=int)
        for i, c in enumerate(channels_numbers):
            inverse_map[c] = i
        return (
            np.array(self.measurements["dyn"], dtype=int),
            inverse_map[np.array(self.measurements["chan"])],
            np.array(self.measurements["ky"], dtype=int)
            )

    def get_cardiac_bins_channel_indices_and_kyindices(self):
        """
        Returns the index k of the cardiac bin, the channel c (in 0...C-1), and the ky index of all measurements in chronological order.
        """
        channels_numbers = self.get_channel_numbers()
        inverse_map = np.zeros(max(channels_numbers)+1, dtype=int)
        for i, c in enumerate(channels_numbers):
            inverse_map[c] = i
        return (
            np.array(self.measurements["card"], dtype=int),
            inverse_map[np.array(self.measurements["chan"])],
            np.array(self.measurements["ky"], dtype=int)
            )
        
    def get_dynamics_and_kyindices(self):
        """
        Returns the index k of the dynamic and the ky index of all measurements in chronological order where channels have been grouped.
        """
        dynamics, channels, ky_indices = self.get_dynamics_channel_indices_and_kyindices()
        dynamics = dynamics[channels == 0]
        ky_indices = ky_indices[channels == 0]

        return dynamics, ky_indices

    def get_number_of_ky_lines(self):
        _, channels, ky_indices = self.get_dynamics_channel_indices_and_kyindices()
        ky_indices = ky_indices[channels == 0]
        return len(ky_indices)

    def get_number_of_dynamics(self):
        return len(np.array(self.measurements["dyn"].unique()))


class PhysLogData():
    """
    Data loader/parser and utils for PhysLog files.
    
    Example usage:
    ```python
    metadataset_name = "ci_01032022_2018056_9_1_wip_csbtfe_freebreathingV4_SCANPHYSLOG"
    metadata_file_name = "/root/export/{}.log".format(metadataset_name)
    physlogData = PhysLogData(metadata_file_name)
    cardiac_phases, cardiac_cycles = physlogData.cardiac_phase_and_cycle(index_range=slice(6000, None))
    physlogData.plotECGAndRespiratory()
    ```
    """
    def __init__(self, file_name):

        # Read physlog file with PANDA
        raw_data=pd.read_csv(file_name, sep=' ',skiprows=6, header=None, names=['v1raw','v2raw','v1','v2','ppu','resp','gx','gy','gz','mark1','mark2'], dtype={'ID':np.str}, skipinitialspace=True)

        # Create a time vector using the known dwell time
        dwell = 1.0 / 496.0 # Physiology dwell time (clock rate is 496 Hz)
        numSamples = len(raw_data)
        time = np.arange(numSamples) * dwell

        # Init a dictionary of lists
        data = self.initDictOfLists()

        # Fill the traces directly from the data matrix
        data["v1raw"] = raw_data.v1raw
        data["v2raw"] = raw_data.v2raw
        data["v1"] = raw_data.v1
        data["v2"] = raw_data.v2
        data["ppu"] = raw_data.ppu
        data["resp"] = raw_data.resp
        data["gx"] = raw_data.gx
        data["gy"] = raw_data.gy
        data["gz"] = raw_data.gz
        data["time"] = time
        data["dwell"] = dwell


        # Converts a number (0,2,4,8) to a mask of 4 bits
        def toBitmask(intNumber):
            width = 4 #4bit width
            output = [int(x) for x in '{:0{size}b}'.format(intNumber,size=width)]
            return(output)
            
        # Convert a marker as read from the physlog file to a 4-digit string. As these may be recognised as numbers or strings, we need a few if/elses
        def convertTo4DigitStr(marker):   
            if np.issubdtype(type(marker), np.integer):
                intNumber = int(marker)
            
            if type(marker) is str:
                if marker.isdigit():
                    intNumber = int(marker)
                else:
                    intNumber = 0   
            digitString = '{:04.0f}'.format(intNumber)
            return(digitString)

        # Decode the two markers into a 32 bitmask, and add the markers to the appropriate list (at the respective entry in the dictionary)
        def decodeMarkers(mark1, mark2, curDict):
            bitmask = []
            # Concat the two markers (as strings)
            digitString = convertTo4DigitStr(mark2) + convertTo4DigitStr(mark1)
            #Convert into bitmask
            for item in digitString:
                word = toBitmask(int(item))
                bitmask += word
            
            bitmask.reverse()    
            # Store the bits in the appropriate marker meaning   
            for bitNr, val in enumerate(bitmask):
                curDict[self.markerBits[bitNr]].append(val)
                
            return(curDict)
        
        # Decode the markers, and add them to our dictionary
        for idx in range(0, numSamples):
            mark1 = raw_data.mark1[idx]
            mark2 = raw_data.mark2[idx]
            decodeMarkers(mark1, mark2, data)
        
        self.data = data

    ## Getters and Analysis tools
    def plotECGAndRespiratory(self):
        """ 
        Plot the ECG trace, respiratory trace, measurement markers, ECG triggers, and scan begin/end markers in a figure.
        """
                
        # Create a plot
        f=plt.figure(figsize=(400,20))        
        ax = f.add_subplot(211)
        ax.plot(self.data["time"], self.data["v1raw"] )

        dwell = self.data["dwell"]

        # Plot the marker for ECG trigger points as a black x over the ECG
        for ECGTriggerPoint in self.ecg_trigger_indices():
            ax.plot(ECGTriggerPoint*dwell, self.data["v1raw"][ECGTriggerPoint], 'bx')

        # Plot the marker for scan start as a red circle over the ECG
        for scanBegin in self.scan_begin_indices():
            ax.plot(scanBegin*dwell, self.data["v1raw"][scanBegin], 'ro')
            
        # Plot the marker for scan end as a green circle over the ECG
        for scanEnd in self.scan_end_indices():
            ax.plot(scanEnd*dwell, self.data["v1raw"][scanEnd], 'go')
            
        # Plot the respiratory curve
        ax = f.add_subplot(212)
        ax.plot(self.data["time"], self.data["resp"] )

        # Plot the measurement markers as blue circles over the respiration curve
        for measurement in self.measurement_indices():
            ax.plot(measurement*dwell, 0, 'bo')
        return f

    def measurement_indices(self):
        """
        Returns the indices with measurement markers.
        """
        return np.nonzero(self.data["MEASUREMENT"])[0]
    
    def ecg_trigger_indices(self):
        """
        Returns the indices with ecg trigger markers.
        """
        return np.nonzero(self.data["TRIGGER_ECG"])[0]
    
    def scan_begin_indices(self):
        """
        Returns the indices with scan begin markers.
        """
        return np.nonzero(self.data["SCAN_BEGIN"])[0]

    def scan_end_indices(self):
        """
        Returns the indices with scan end markers.
        """
        return np.nonzero(self.data["SCAN_END"])[0]

    def cardiac_phase_and_cycle(self, index_range=None):
        """
        Returns the cardiac phase and cycle for all measurement markers.
        """
        if isinstance(index_range, slice):
            measurement_indices = np.nonzero(self.data["MEASUREMENT"][index_range])[0]
            ecg_triggers_indices = np.nonzero(self.data["TRIGGER_ECG"][index_range])[0]
        else:
            measurement_indices = self.measurement_indices()
            ecg_triggers_indices = self.ecg_trigger_indices()

        Nk = len(measurement_indices)
        Nct = len(ecg_triggers_indices)

        # find cardiac phase in range [0, 1) of each measurement: linearly interpolate between ECG triggers
        # and find index of the heart beat for each measurement
        cardiac_cycles = np.zeros((Nk))
        cardiac_phases = np.zeros((Nk))
        k = 0
        for c in range(Nct-1):
            previous = ecg_triggers_indices[c]
            next = ecg_triggers_indices[c+1]
            while k < Nk and previous <= measurement_indices[k] < next:
                cardiac_phases[k] = (measurement_indices[k] - previous) / (next - previous) # linear interpolation
                cardiac_cycles[k] = c
                k += 1

        cardiac_cycles -= cardiac_cycles[0]

        return cardiac_phases, cardiac_cycles

    ## Helper methods
    # Define a dictionary, where each dictionary entry is a list of the samples (or markers)
    def initDictOfLists(self):
        curDict = {}
        for markerCode in self.markerBits:
            curDict[markerCode] = []
        return( curDict )

    def single_marker_cardiac_and_respiratoy_info(self, t_period=None, num_samples=None, sample_times=None):
        """ 
        Returns the cardiac phase and respiratory state for measurements that were started after the first measurement marker and are recorded in regular intervals t_period.
        """
        # find the measurement marker and check if the scan was started before
        measurement_indices = self.measurement_indices()
    
        # print(measurement_indices)
        # print(self.scan_begin_indices())

        # assert len(measurement_indices) == 1, "Invalid PhyslogFile format, there must only be one measurement marker."
        if len(measurement_indices) > 1:
            print("Warning: Multiple measurement markers are contained in the physlog. Using first one. Good luck.")

        first_measurement_index = measurement_indices[0]

        scan_begin_indices = self.scan_begin_indices()
        assert (scan_begin_indices <= first_measurement_index).any()

        if t_period is not None and num_samples is not None:
            sample_times = np.arange(num_samples) * t_period
        elif sample_times is not None:
            sample_times = np.array(sample_times)
            num_samples = len(sample_times)
        else:
            raise Exception

        ecg_states = np.zeros(num_samples)
        respiratory_states = np.zeros(num_samples)
        cardiac_phases = np.zeros(num_samples)
        cardiac_cycles = np.zeros(num_samples)

        def linear_interpolation(a, b, fraction):
            return a * fraction + b * (1 - fraction)

        ecg_trace = self.data["v1raw"]
        respiratory_trace = self.data["resp"]

        ecg_trigger_indices = self.ecg_trigger_indices()


        for n, t in enumerate(sample_times):
            fractional_index = t / self.data["dwell"] + first_measurement_index
            last_index = np.floor(fractional_index) 
            next_index = np.ceil(fractional_index)
            fraction = fractional_index % 1. 

            assert fractional_index < len(ecg_trace), "Specified (t_period*num_samples) or sample_times exceeds the duration of the Physlog."

            # interpolate states linearly
            ecg_states[n] = linear_interpolation(ecg_trace[last_index], ecg_trace[next_index], fraction)
            respiratory_states[n] = linear_interpolation(respiratory_trace[last_index], respiratory_trace[next_index], fraction)
            
            # find position of previous and next ecg trigger index

            previous_ecg_trigger_index = -1
            for i in range(len(ecg_trigger_indices)-2, -1, -1):
                if ecg_trigger_indices[i] <= fractional_index:
                    previous_ecg_trigger_index = i
                    break
            assert previous_ecg_trigger_index != -1

            cardiac_cycles[n] = previous_ecg_trigger_index
            cardiac_phases[n] = (fractional_index - ecg_trigger_indices[previous_ecg_trigger_index]) / (ecg_trigger_indices[previous_ecg_trigger_index+1] - ecg_trigger_indices[previous_ecg_trigger_index])

        cardiac_cycles -= cardiac_cycles[0]

        return {
            "ecg_states": ecg_states,
            "respiratory_states": respiratory_states,
            "cardiac_cycles": cardiac_cycles,
            "cardiac_phases": cardiac_phases
        }

    ## Static attributes
    markerBits = ["TRIGGER_ECG",    # 0 --> 0001 0000
              "TRIGGER_PPU",        # 1 --> 0002 0000
              "TRIGGER_RES",        # 2 --> 0004 0000
              "MEASUREMENT",        # 3 --> 0008 0000
              "SCAN_BEGIN",         # 4 --> 0010 0000
              "SCAN_END",           # 5 --> 0020 0000
              "TRIGGER_EXTERNAL",   # 6 --> 0040 0000
              "CAL_AUTO",           # 7 --> 0080 0000 cal data end
              "MANUAL_START",       # 8 --> 0100 0000
              "CAL_MANUAL_OUT",     # 9 --> 0200 0000 cal data end
              "CAL_MANUAL_IN",      #10 --> 0400 0000 cal data end RC: cal A or cal C
              "NOTUSED_11",         #11 --> 0800 0000              RC: cal B or cal B
              "NOTUSED_12",         #12 --> 1000 0000 RC patch: act A or act C
              "NOTUSED_13",         #13 --> 2000 0000 RC patch: act B or act C
              "NOTUSED_14",         #14 --> 4000 0000 RC patch: new algorithm
              "NOTUSED_15",         #15 --> 8000 0000 after this start 2nd word: RC: old algorithm 
              "VCG_TRIGGER_AUTO",   #16 --> 0000 0001 Trigger based on Auto
              "VCG_TRIGGER_MAN_OUT",#17 --> 0000 0002 Trigger based on ManOut
              "VCG_TRIGGER_MAN_IN", #18 --> 0000 0004 Trigger based on ManIn
              "NOTUSED_19",         #19 --> 0000 0008 not used
              "ACTIVE_CAL_AUTO",    #20 --> 0000 0010 calibration used by analyze
              "ACTIVE_CAL_MAN_OUT", #21 --> 0000 0020 idem
              "ACTIVE_CAL_MAN_IN",  #22 --> 0000 0040 idem
              "NOTUSED_23",         #23 --> 0000 0080 not used
              "TRIGLEVEL_LOW",      #24 --> 0000 0100 used triggerlevel low
              "TRIGLEVEL_DEFAULT",  #25 --> 0000 0200 idem              default
              "TRIGLEVEL_HIGH",     #26 --> 0000 0400 idem              high
              "NOTUSED_27",         #27 --> 0000 0800 not used
              "SIMULATED_DATA",     #28 --> 0000 1000 Simulated physiology data
              "SENSOR_DISCONNECTED",#29 --> 0000 2000 Sensor got disconnected
              "NOTUSED_30",         #30 --> 0000 4000 not used
              "NOTUSED_31"]         #31 --> 0000 8000 not used

class XCATLog():
    """
    Parser for XCAT log files for reading the cardiac and respiratory phase of the phantom images.
    """
    def __init__(self, file_name):

        def parseVariable(line, string_pre_number, number_len):
            if line.startswith(string_pre_number):
                n = len(string_pre_number)
                return [float(line[n:(n+number_len)])]
            return []

        heart_phases = []
        respiratory_phases = []
        heart_cycles = []

        # Read log_file line by line
        with open(file_name) as file:
            for line in file:
                line_txt = line.rstrip()

                heart_phases += parseVariable(line_txt, "Current heart phase index =   ", 5)
                respiratory_phases += parseVariable(line_txt, "Current resp phase index  =   ", 5)

        last = 0
        cycle = 0
        for heart_phase in heart_phases:
            if last > heart_phase:
                cycle += 1
            last = heart_phase
            heart_cycles.append(cycle)

        self.heart_phases = heart_phases
        self.respiratory_phases = respiratory_phases
        self.heart_cycles = heart_cycles




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


class ExamCard():
    """ 
    Class for parsing and accessing information from examcard *.txt files.
    
    Each line of the examcard file is available as an entry in the `examcard.parameters` dict. The keys are the text on the left hand side of the `=` sign with leading and trailing whitespaces removed. If multiple lines share the same left hand side, the keys are extended by numbers `2, 3, ...`.
    """
    def __init__(self, examcard_path):
        with open(examcard_path) as f:
            self.raw = f.readlines()

        self.parameters = dict()
        
        def parse_value(text):
            if text == "\"yes\"":
                return True
            elif text == "\"no\"":
                return False
            elif re.match("\"([0-9]*.[0-9]*) \/ ([0-9]*.[0-9]*)\"",text) or re.match("\"([0-9]*.[0-9]*) \/ ([0-9]*.[0-9]*) \/ ([0-9]*.[0-9]*)\"",text):
                seg = [float(t) for t in text[1:-1].split("/")]
                return tuple(seg)
            elif text.startswith("\"") and text.endswith("\""):
                return text[1:-1]
            elif text.isdigit():
                return int(text)
            elif text.replace('.', '', 1).isdigit():
                return float(text)
            else:
                try:
                    return float(text)
                except:
                    return text

        for line in self.raw:
            seg = line.replace("\n", "").replace("\t","").replace(";","").split("=")
            key = "=".join(seg[0:-1]).lstrip().rstrip()
            value = seg[-1].lstrip().rstrip()

            if key in self.parameters.keys():
                i = 2
                key_new = "{}_{}".format(key, i)
                while key_new in self.parameters.keys():
                    i += 1
                    key_new = "{}_{}".format(key, i)
                key = key_new
                   
            self.parameters[key] = parse_value(value)

    @staticmethod
    def count_leading_whitespaces(text):
        return len(text) - len(text.lstrip())


def mask_to_trajectory(mask, Nz, Ny, Nx):
    trajectory = mask.clone().type(dtype)
    trajectory[..., 0] = 2. * torch.pi / Nz * (-int(Nz/2) + trajectory[..., 0])
    trajectory[..., 1] = 2. * torch.pi / Ny * (-int(Ny/2) + trajectory[..., 1]) 
    trajectory[..., 2] = 2. * torch.pi / Nx * (-int(Nx/2) + trajectory[..., 2]) 
    return trajectory