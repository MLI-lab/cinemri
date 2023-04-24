# Implicit Neural Networks with Fourier-Feature Inputs for Free-breathing Cardiac MRI Reconstruction

## Updates

## Setup
1. Setup a docker container with support for Nvidia GPUs and pytorch. 
2. Install additional packages
```bash
cd ./setup
chmod +x setup.sh
./setup.sh
```

## Datasets

## Licence


## Supplementary materials
### Reconstructed videos

#### Low-resolution high-SNR dataset
The video below shows the reconstructions of the low-resolution high-SNR dataset by the FMLP, the KFMLP, and the t-DIP for an acquisition time of $4s$ ($T = 225$). An ECG-gated breath-hold (BH) dataset was reconstructed using classical sparsity-based methods and is shown as visual reference. It can be seen that the reconstruction quality of the FMLP and the t-DIP are on par, whereas the KFMLP suffers aliasing-like artifacts. 

![Reconstructions of the low-resolution high-SNR dataset with the FMLP, KFMLP, and t-DIP.](supplements/lowres_highsnr/all_methods_225/timecoded_cfr.gif)


##### Reconstructions by the FMLP using different acquisition lengths
For the video below, the FMLP was trained on three different acquisition lengths $4s$ ($T=225$), $8s$ ($T=512$), and $16s$ ($T=900$). It can be seen that the visual reconstruction quality does not improve notably with increased acquistion time.

![FMLP reconstructions of the low-resolution high-SNR dataset with different acquisition lengths](supplements/lowres_highsnr/FMLP/all_acquisition_lengths/timecoded_cfr.gif)


#### Low-resolution low-SNR dataset
The low-resoltion low-SNR dataset was reconstructed by the FMLP, the KFMLP, and the t-DIP for an acquisition time of $4s$ ($T = 225$). It can be seen below that the reconstruction quality of the FMLP and the t-DIP is similar and both methods exhibit similar artifacts. The KFMLP, by contrast, exhibits aliasing-like artifacts and high-frequency noise. 

![Reconstructions of the low-resolution low-SNR dataset with the FMLP, KFMLP, and t-DIP.](supplements/lowres_lowsnr/all_methods_225/timecoded_cfr.gif)

#### High-resolution dataset
The reconstructions of high-resolution dataset for an acquisition time of $4s$ ($T=225$) are depricted below. Again, the FMLP and the t-DIP are similar in image quality. The KFMLP's reconstruction is obscured substantially by noise.

![Reconstructions of the high-resolution dataset with the FMLP, KFMLP, and t-DIP.](supplements/highres/all_methods_225/timecoded_cfr.gif)
