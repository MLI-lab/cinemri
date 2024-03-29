"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)

def to_numpy(data):
    if torch.is_complex(data) or data.shape[-1] != 2:
        return data.numpy()
    return data[...,0].numpy() + 1j * data[...,1].numpy()

def apply_mask(data, mask_func = None, mask = None, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    if mask is None:
        mask = mask_func(shape, seed)
    return data * mask, mask


def mask_center(x, mask_from, mask_to):
    b, c, h, w, two = x.shape
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
    return mask

def complex_mul(x, y):
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)


def complex_conj(x):
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)

'''
def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data
'''

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()

def complex_abs_np(data): ############################################### MZD
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.shape[-1] == 2
    return np.sqrt( (data ** 2).sum(axis=-1) )

def complex_abs_sq(data):
    """
    Compute the squared absolute value of a complex tensor
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1)


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))

def root_sum_of_squares2(lsimg):
    out = np.zeros(lsimg[0].shape)
    for img in lsimg:
        out += img**2
    return np.sqrt(out)

def root_sum_of_squares_complex(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))

def channels2imgs(out):
    sh = out.shape
    chs = int(sh[0]/2)
    #imgs = np.zeros( (chs,sh[1],sh[2]) )
    imgs = np.zeros( (chs,*sh[1:]) )
    for i in range(chs):
        imgs[i] = np.sqrt( out[2*i]**2 + out[2*i+1]**2 )
    return imgs

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimension should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(x, y):
    """
    Apply a center crop on the larger image to the size of the smaller image.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))
    return x, y


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.
        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero
        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def fft(input, signal_ndim, normalized=False):
    # This function is called from the fft2 function below
    if signal_ndim < 1 or signal_ndim > 3:
        print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
        return

    dims = (-1)
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)

    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(torch.fft.fftn(torch.view_as_complex(input), dim=dims, norm=norm))

def ifft(input, signal_ndim, normalized=False):
    # This function is called from the ifft2 function below
    if signal_ndim < 1 or signal_ndim > 3:
        print("Signal ndim out of range, was", signal_ndim, "but expected a value between 1 and 3, inclusive")
        return

    dims = (-1)
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)

    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(torch.fft.ifftn(torch.view_as_complex(input), dim=dims, norm=norm))

def fft1(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 1 dimensional Fast Fourier Transform. It calls the fft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 2 dimensions: dimensions
            -2 is the spatial dimension and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-2))
    data = fft(data, 1, normalized=True)
    data = fftshift(data, dim=(-2))
    return data


def ifft1(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 1-dimensional Inverse Fast Fourier Transform. It calls the ifft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 2 dimensions: dimensions
            -2 is the spatial dimension and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-2))
    data = ifft(data, 1, normalized=True)
    data = fftshift(data, dim=(-2))
    return data

def fft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2 dimensional Fast Fourier Transform. It calls the fft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 2-dimensional Inverse Fast Fourier Transform. It calls the ifft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def fft3(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 3 dimensional Fast Fourier Transform. It calls the fft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-4, -3, -2))
    data = fft(data, 3, normalized=True)
    data = fftshift(data, dim=(-4, -3, -2))
    return data


def ifft3(data):
    """
    ref: https://github.com/facebookresearch/fastMRI/tree/master/fastmri
    Apply centered 3-dimensional Inverse Fast Fourier Transform. It calls the ifft function above to make it compatible with the latest version of pytorch.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-4, -3, -2))
    data = ifft(data, 3, normalized=True)
    data = fftshift(data, dim=(-4, -3, -2))
    return data