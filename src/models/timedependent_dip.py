from cmath import isnan
from ctypes import ArgumentError
from random import random
import torch
from torch import nn
import torchkbnufft as tkbn

import numpy as np
import torchvision

import matplotlib.pyplot as plt
import time

from torch.utils.tensorboard import SummaryWriter

from src import *

### layers ###
class ConvBNReLU(nn.Module):
    """
    Conv -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, bn_affine=True):
        super().__init__()

        self.net = nn.Sequential()
        self.net.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        self.net.append(nn.BatchNorm2d(out_channels, affine=bn_affine))
        self.net.append(nn.ReLU())
    
    def forward(self, x):
        return self.net(x)

class NNConvBNReLU(nn.Module):
    """
    NN-Interpolation -> `num_conv` x (Conv -> BatchNorm -> ReLU)
    """
    def __init__(self, channels, out_size, num_conv, bias=False, bn_affine=True):
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(nn.Upsample(size=out_size, mode="nearest"))
        for i in range(num_conv):
            self.net.append(ConvBNReLU(channels, channels, bias=bias, bn_affine=bn_affine, kernel_size=3, stride=1, padding=1))
    
    def forward(self, x):
        return self.net(x)

class MapNet(nn.Module):
    def __init__(self, in_features, out_size, width=512, hidden_layers=1):
        super().__init__()

        self.out_size = out_size

        self.net = nn.Sequential()
        self.net.append(nn.Linear(in_features, width, bias=True))
        self.net.append(nn.ReLU())

        for i in range(1, hidden_layers):
            self.net.append(nn.Linear(width, width, bias=True))
            self.net.append(nn.ReLU())
        
        self.net.append(nn.Linear(width, out_size[0]*out_size[1], bias=True))
    
    def forward(self, x):
        return self.net(x).reshape(-1, 1, self.out_size[0], self.out_size[1]) # B C W H

class Decoder(nn.Module):
    def __init__(self, in_features, out_features, out_size, map_net_out_size, num_stages, num_conv, conv_channels, conv_bias, output_scaling):
        """
        Decoder: MapNet -> CNN -> Crop

        Parameters:
        - `in_features`: number of input features to MapNet
        - `out_features`: number of output features/channels of the CNN
        - `out_size`: image resolution at the output of the crop
        - `map_net_out_size`: output feature resolution of MapNet = input feature resolution of the CNN
        - `num_stages`: number of interpolation stages in the CNN
        - `num_conv`: number of convolutions between NN-interpolation layers
        - `conv_channels`: number of channels of the hidden convolutional layers
        - `conv_bias`: sets the bias of the convolutional layers
        """
        super().__init__()

        # check that the output resultion of the CNN is at least as large as the image resolution (if not, `num_stages` or  `map_net_out_size` needs to be adjusted).
        final_size = (map_net_out_size[0]* 2**(num_stages), map_net_out_size[1]* 2**(num_stages))
        assert final_size[0] >= out_size[0] and final_size[1] >= out_size[1], "The output resolution if the CNN is too small."

        self.map_net = MapNet(in_features, map_net_out_size, width=512, hidden_layers=1)

        intermediate_sizes = []
        for i in range(num_stages):
            intermediate_sizes.append((map_net_out_size[0]* 2**(i+1), map_net_out_size[1]* 2**(i+1)))
        print(intermediate_sizes)

        self.output_scaling = output_scaling

        # CNN
        self.net = nn.Sequential()
        self.net.append(ConvBNReLU(1, conv_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias, bn_affine=True))
        for i in range(num_stages):
            self.net.append(NNConvBNReLU(conv_channels, intermediate_sizes[i], num_conv=num_conv, bias=conv_bias, bn_affine=True))
        
        self.net.append(nn.Conv2d(conv_channels, out_features, kernel_size=3, stride=1, padding=1, bias=conv_bias))
        self.net.append(torchvision.transforms.CenterCrop(out_size)) # center-crop to the final imageresolution

    def forward(self, x, return_map=False):
        embedded = self.map_net(x)
        out = self.net(embedded) * self.output_scaling

        if return_map:
            return out, embedded

        return out


class ReconstructionMethod():
    def __init__(self, param):
        self.param = param

        self.decoder = Decoder(**vars(param.decoder)).type(dtype)

        if param.trajectory.type == "helix":
            self.trajectory = self.init_helix_trajectory().type(dtype)
        
        parameters = [x for x in self.decoder.parameters()]
        self.optimizer = torch.optim.Adam(parameters, **vars(self.param.optimizer))

        self.metrics = PerformanceMetrics(**vars(param.metrics))

        self.forward_operator = ForwardOperator(dataset_type=self.param.data.dataset_type, Ny=param.data.Ny, Nx=param.data.Nx).type(dtype)
    
    def init_helix_trajectory(self):
        assert len(self.param.trajectory.z_slack) == self.param.trajectory.L - 2

        if self.param.trajectory.equal_frame_size:
            # compute trajectory based on sample indices k -> assumes that all frames are of equal size
            k = torch.arange(0, self.param.data.Nk, step=1) / (self.param.data.Nk - 1)
        else:
            # compute trajectory based on the frame times t_k that may be spaced unevenly
            k = torch.tensor(self.param.data.frame_times[0:self.param.data.Nk]).clone()
            k = k - k[0]
            k = k / k[-1]
            assert k[0] == 0. and k[-1] == 1.

        circle = torch.stack((
            torch.cos(2*torch.pi*self.param.trajectory.p*k),
            torch.sin(2*torch.pi*self.param.trajectory.p*k)
        ), dim=-1)

        linear = torch.tensor(self.param.trajectory.z_slack).unsqueeze(dim=0)*k.unsqueeze(dim=1)

        return torch.concat((circle, linear), dim=1)

    def load_state(self, path, gpu):
        states = torch.load(path, map_location=torch.device('cuda', gpu))
        self.decoder.load_state_dict(states["decoder_state_dict"])
        self.optimizer.load_state_dict(states["optimizer"])

    def save_state(self, path):
        torch.save({
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def reconstruction_loss(self, img, sample, smaps):
        kspace_hat = self.forward_operator(img, sample, smaps=smaps)

        squared_errors = torch.square(kspace_hat - sample["kspace"])
        squared_errors = torch.sum(torch.mean(squared_errors, dim=0)) # mean over batch, sum over rest

        return squared_errors


    def evaluate(self, indices, return_latent=False):
        if isinstance(indices, int):
            indices = [indices]

        for i in indices:
            assert i >= 0 and i < self.trajectory.shape[0], "index {} out of range".format(i)

        img, mapped = self.decoder(self.trajectory[indices, :], return_map=True)
        img = img.permute((0, 2, 3, 1))

        if return_latent:
            return img, mapped
        else:
            return img
        
    
    def evaluate_abs(self, indices):
        img = self.evaluate(indices)
        return root_sum_of_squares(img, dim=-1) # compute magnitude image or RSS image

    def evaluate_npy(self, k):
        return self.evaluate_abs(k).detach().cpu().numpy().squeeze()

    def plot_latent_space_pca(self, dataset, colors=None, add_sample_indices=False):

        mapped_size = self.param.decoder.map_net_out_size[0] * self.param.decoder.map_net_out_size[1]
        # compute all means and logvars
        latent_mus = np.zeros((self.param.data.Nk, mapped_size))

        for k in range(self.param.data.Nk):
            img, mu = self.evaluate(k, return_latent=True)
            img = img.detach().cpu()
            latent_mus[k,:] = mu.detach().cpu().numpy().flatten()

        # scatter plot of the latent mus with sigma ellipses
        embedded_coords, V = pca(latent_mus, return_v=True)

        if colors is None:
            colors = np.zeros(self.param.data.Nk)

        fig = plt.figure(figsize=(10,10))
        ax1 = fig.add_subplot(111)
        ax1.set_title("Two principal components of mapnet output")
        ax1.scatter(embedded_coords[:,0], embedded_coords[:,1], c=colors[0:self.param.data.Nk])

        # add labels
        if add_sample_indices:
            for itxt in range(self.param.data.Nk):
                ax1.annotate(str(itxt), (embedded_coords[itxt,0], embedded_coords[itxt,1]), fontsize=7)

        return fig


    def train(self, dataset, validation_dataset=None):
        # setup logging of the traing
        self.writer = SummaryWriter(log_dir=self.param.experiment.results_dir)

        # create a directory for models and intermediate images/pca analysis during training
        training_dir = os.path.join(self.param.experiment.results_dir, "training")
        create_dir(training_dir)

        # use the torch DataLoader for loading random mini-batches and memory optimization
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collocate_concat)
        
        smaps = dataset.smaps.squeeze(dim=1).type(dtype)

        img = self.evaluate_npy(0)
        self.writer.add_image("initial img", img / np.max(img), 0, dataformats="HW")

        print("Starting training... result directory: {}".format(self.param.experiment.results_dir))

        max_ser = 0.
        max_ser_epoch = 0
        max_ser_subset = 0.
        validation_results = {}

        max_ssim = 0.
        max_vif = 0.
        i = 1
        num_epochs = self.param.hp.num_iter
        while i <= num_epochs: # iterate over epochs

            start_time = time.time()
            
            loss_avg = 0.
            loss_reconstruction_avg = 0.
            num_samples = 0.

            for sample in self.dataloader: # iterate over frames
                sample = copySampleToGPU(sample)

                self.optimizer.zero_grad()
                
                img = self.evaluate(sample["indices"])

                loss_reconstruction = self.reconstruction_loss(img, sample, smaps)

                # total loss
                loss = loss_reconstruction

                loss.backward()
                self.optimizer.step()

                loss_reconstruction_avg += loss_reconstruction.detach()
                loss_avg += loss.detach()
                num_samples += 1 # kld and reconstruction loss is already averaged over the batch


            loss_reconstruction_avg /= num_samples
            loss_avg /= num_samples
            
            
            if i%self.param.experiment.model_save_frequency == 0 or i==self.param.hp.num_iter:
                # save intermediate model parameters and optimizer state
                self.save_state(os.path.join(training_dir, "trained_model_{}.pth".format(i)))

            if i%self.param.experiment.evaluation_frequency == 0 or i==self.param.hp.num_iter or i == 1:
                # reconstruction of the first frame, crosssection and latent space pca analyis
                img = self.evaluate_npy(100)
                self.writer.add_image("performance/frame_100", img / np.max(img), i, dataformats="HW")

                # plot of the latent space pca
                fig_pca = self.plot_latent_space_pca(dataset, colors=self.param.data.cardiac_phases, add_sample_indices=True)
                self.writer.add_figure("latent_space/pca", fig_pca, i)
            

            if dataset.reference is not None and self.param.experiment.evaluate_reference_metrics and (i%self.param.experiment.reference_evaluation_frequency == 0 or i==self.param.hp.num_iter-1 or i == 1):
                self.metrics.clear()
                for k in self.param.data.sample_indices:
                    img = self.evaluate_abs(k).detach()
                    self.metrics.add(img.squeeze(), torch.tensor(dataset.reference[k, 0, :, :]))
                self.metrics.save_all_to_history(i)
                self.metrics.save_history_to_file(os.path.join(training_dir, "metrics.pth"))
                mean_ssim = np.mean(np.array(self.metrics.history["ssim"][-1]))
                mean_vif = np.mean(np.array(self.metrics.history["vif"][-1]))
                self.writer.add_scalar('performance/ssim', mean_ssim, i)
                self.writer.add_scalar('performance/vif', mean_vif, i)
                max_ssim = max(max_ssim, mean_ssim)
                max_vif =  max(max_vif, mean_vif)
                self.writer.add_scalar('performance/max_ssim', max_ssim, i)
                self.writer.add_scalar('performance/max_vif', max_vif, i)
                self.metrics.clear()

            if i%self.param.experiment.video_evaluation_frequency == 0 or i==self.param.hp.num_iter-1 or i == 1:
                imgs = torch.stack([self.evaluate_abs(k).detach().cpu() for k in self.param.data.sample_indices], dim=0)
                imgs /= torch.max(imgs)
                imgs = imgs.unsqueeze(dim=0) # format to: B=1, N, C=1, H, W
                self.writer.add_video("video", imgs, i, self.param.data.frame_rate)
            
            if validation_dataset is not None and (i%self.param.experiment.validation_evaluation_frequency == 0 or i==self.param.hp.num_iter-1 or i == 1):
                ser, ser_subset, squared_errors_and_signal_power = self.evaluate_validation(validation_dataset, smaps)
                
                validation_results[i] = squared_errors_and_signal_power
                torch.save(validation_results, os.path.join(training_dir, "validation_results.pth"))

                self.writer.add_scalar('performance/validation_ser', ser, i)
                self.writer.add_scalar('performance/validation_ser_subset', ser_subset, i)
                
                if max_ser < ser:
                    max_ser = ser
                    max_ser_epoch = i
                    self.save_state(os.path.join(training_dir, "ser_highscore.pth"))

                if max_ser_subset < ser_subset:
                    max_ser_subset = ser_subset
                    self.save_state(os.path.join(training_dir, "ser_subset_highscore.pth"))

                if self.param.hp.extend_training_until_no_new_ser_highscore:
                    num_epochs = max(num_epochs, max_ser_epoch + self.param.hp.num_epochs_after_last_highscore)
        
                self.writer.add_scalar('performance/max_validation_ser', max_ser, i)
                self.writer.add_scalar('performance/max_validation_ser_subset', max_ser_subset, i)
                
            stop_time = time.time()
            eta = (stop_time - start_time)*(num_epochs - (i-1))/60

            self.writer.add_scalar('train/loss', loss_avg, i)
            self.writer.add_scalar('train/loss_reconstruction', loss_reconstruction_avg, i)

            print("Iteration {}: Train loss {:.7f}, reconstruction: {:.7f}, eta: {:.1f}min".format(i, loss_avg, loss_reconstruction_avg, eta), '\r', end='')
            i += 1

        print("\n") # clear stdout
                
    def evaluate_metrics(self, dataset, iteration, clear_afterwards=True):
        self.metrics.clear()
        for k, sample in enumerate(dataset):
            img = torch.tensor(self.evaluate_abs(k).squeeze())
            self.metrics.add(img, sample["reference"])
        self.metrics.save_all_to_history(iteration)
        if clear_afterwards:
            self.metrics.clear()

    
    def evaluate_validation(self, dataset, validation_dataset, smaps):
        squared_errors = []
        squared_signals = []
        validation_line_indices = []

        squared_errors_validation_subset = []
        squared_signals_validation_subset = []

        for v in validation_dataset:

            is_in_sample_indices = v["k"] in self.param.data.sample_indices
            is_in_subset_indices = v["line_index"] in self.param.experiment.validation_subset_indices

            if is_in_subset_indices and (not is_in_sample_indices):
                print("error:")
                print(v)
                raise Exception

            if not is_in_sample_indices:
                continue

            v = copySampleToGPU(v)
            img = self.evaluate(v["k"])
            img = img.detach()

            if "mask" in v.keys(): # cartesian dataset
                kspace_rec = self.forward_operator.forward_sparse_cartesian(img, smaps, v["mask"])
            else:
                kspace_rec = self.forward_operator.forward_non_cartesian(img, smaps, v["trajectory"])

            se = torch.sum(torch.square(kspace_rec - v["kspace"])).detach().cpu().numpy()
            squared_signal = torch.sum(torch.square(v["kspace"])).detach().cpu().numpy()
            
            if is_in_subset_indices:
                squared_errors_validation_subset.append(se)
                squared_signals_validation_subset.append(squared_signal)

            squared_errors.append(se)
            squared_signals.append(squared_signal)
            validation_line_indices.append(v["line_index"])

        mse = np.mean(np.array(squared_errors))
        mean_power = np.mean(np.array(squared_signals))
        ser = 10 * np.log10(mean_power / mse)

        mean_power_subset = np.mean(np.array(squared_signals_validation_subset))
        mse_subset = np.mean(np.array(squared_errors_validation_subset))
        ser_subset = 10 * np.log10(mean_power_subset / mse_subset)

        squared_errors_and_signal_power = {
            "squared_errors": squared_errors,
            "squared_signals": squared_signals,
            "validation_line_indices": validation_line_indices
        }
        return ser, ser_subset, squared_errors_and_signal_power
    
    def transform(self, sample):

        reference = None
        if sample["reference"] is not None:
            reference = sample["reference"].squeeze()

        # compute the time at the center of the frame
        t_k = float(self.param.data.tr * (sample["line_indices"][0, 0] + sample["line_indices"][0, -1]) / 2)
        
        if "mask" in sample.keys():
            return {
                "kspace": sample["kspace"],
                "mask": sample["mask"],
                "t_k": t_k,
                "indices": sample["indices"],
                "reference": reference
            }
        else:
            return {
                "kspace": sample["kspace"],
                "trajectory": sample["trajectory"],
                "t_k": t_k,
                "indices": sample["indices"],
                "reference": reference
            }