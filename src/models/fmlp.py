from ctypes import ArgumentError
import torch
from torch import nn
import numpy as np
import time

from torch.utils.tensorboard import SummaryWriter

from src import *

### layers ###
class ReluLayer(nn.Module):    
    def __init__(self, in_features, out_features, scale=1, sigma=1, bias=True):
        super().__init__()
        self.scale = scale
        self.sigma = sigma
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.normal_(std=self.sigma)
            if self.linear.bias is not None:
                self.linear.bias.normal_(std=1e-6)

    def forward(self, input):
        tmp = torch.relu(self.linear(input))
        return Normalization(tmp)


class FourierFeatureMap(nn.Module):    
    def __init__(self, in_features, out_features, coordinate_scales):
        super().__init__()

        self.num_freq = out_features // 2
        self.out_features = out_features
        self.coordinate_scales = nn.Parameter(torch.tensor(coordinate_scales).unsqueeze(dim=0))
        self.coordinate_scales.requires_grad = False
        self.linear = nn.Linear(in_features, self.num_freq, bias=False)
        self.init_weights()
        self.linear.weight.requires_grad = False
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.normal_(std=1, mean=0)

    def forward(self, input):
        return torch.cat((np.sqrt(2)*torch.sin(self.linear(self.coordinate_scales*input)), 
                          np.sqrt(2)*torch.cos(self.linear(self.coordinate_scales*input))), dim=-1)
        
### normalization ###
def Normalization(input):  
    if len(input.shape)==3:
        scale = (1/(torch.std(input, unbiased=False, dim=-1) + 1e-5))[:,:,None]
        mean = torch.mean(input,dim=-1)[:,:,None]
    else:
        scale = (1/(torch.std(input, unbiased=False, dim=-1) + 1e-5))[:,None]
        mean = torch.mean(input, dim=-1)[:,None]
    
    return scale*(input-mean)
    

### networks ###
class FMLP(nn.Module):
    def __init__(self,
                spatial_in_features,
                spatial_fmap_width,
                spatial_coordinate_scales,
                
                temporal_in_features,
                temporal_fmap_width,
                temporal_coordinate_scales,

                mlp_width,
                mlp_sigma,
                mlp_scale,
                mlp_hidden_layers,
                mlp_hidden_bias,

                # final layer parameters
                mlp_out_features,
                mlp_final_sigma,
                mlp_final_bias,

                out_scale
                ):
        super().__init__()

        self.spatial_in_features = spatial_in_features
        self.temporal_in_features = temporal_in_features

        self.spatial_fmap = FourierFeatureMap(spatial_in_features, spatial_fmap_width, spatial_coordinate_scales)
        self.temporal_fmap = FourierFeatureMap(temporal_in_features, temporal_fmap_width, temporal_coordinate_scales)

        self.mlp = nn.Sequential()
        self.mlp.append(ReluLayer(spatial_fmap_width + temporal_fmap_width, mlp_width, scale=mlp_scale, sigma=mlp_sigma, bias=mlp_hidden_bias))
        for i in range(1, mlp_hidden_layers):
            self.mlp.append(ReluLayer(mlp_width, mlp_width, scale=mlp_scale, sigma=mlp_sigma, bias=mlp_hidden_bias))
        
        final_linear = nn.Linear(mlp_width, mlp_out_features, bias=mlp_final_bias)
        with torch.no_grad():
            final_linear.weight.normal_(std=mlp_final_sigma)
            if mlp_final_bias:
                final_linear.bias.normal_(std=0.00001)
        self.mlp.append(final_linear)

        self.out_scale = out_scale
    
    def forward(self, coords, temporal_coord=None):

        if temporal_coord is None: # temporal coordinate is part of the coords vector
            spatial_ff = self.spatial_fmap(coords[:, 0:self.spatial_in_features])
            temporal_ff = self.temporal_fmap(coords[:, self.spatial_in_features:])
            combined = torch.concat((spatial_ff, temporal_ff), dim=-1)
        else: # temporal coordinate is provided separately -> more efficient if temporal coordinate is kept constant
            if self.temporal_in_features == 0: # the temporal embedding is not existent -> time is embedded together with spatial coordinates
                temporal_coords = torch.tensor(temporal_coord).type(dtype).unsqueeze(dim=0).repeat(coords.shape[0], 1)
                coords_combined = torch.concat((coords, temporal_coords), dim=-1)
                combined = self.spatial_fmap(coords_combined)
            else:
                spatial_ff = self.spatial_fmap(coords)
                temporal_ff = self.temporal_fmap(temporal_coord)
                combined = torch.concat((spatial_ff, temporal_ff.repeat(spatial_ff.shape[0], 1)), dim=-1)
        
        return self.mlp(combined) * self.out_scale


class TVLoss():
    def __init__(self, num_elements, mode, directionality, normalize):
        self.imgs = dict()
        self.mode = mode
        self.directionality = directionality
        self.num_elements = num_elements
        self.normalize = normalize

    def update_frame(self, k, img):
        assert len(k) == 1
        k = k[0]

        if self.mode == "magnitude":
            assert img.shape[-1] == 2 # check if complex tensor: B W H 2
            self.imgs[k] = torch.sqrt(torch.sum(torch.square(img.detach()), dim=-1)) # compute magnitude image -> B W H
        elif self.mode == "real_imag":
            self.imgs[k] = img.detach()
    
    def compute(self, k, img):

        assert len(k) == 1
        assert len(self.imgs.keys()) == self.num_elements
        k = k[0]

        tv_loss = torch.tensor(0.).type(dtype)

        if self.mode == "magnitude":
            img = torch.sqrt(torch.sum(torch.square(img), dim=-1)) # compute magnitude image -> B W H

        if k != 0:
            tv_loss += torch.sum(torch.abs(img - self.imgs[k - 1])) # TV to previous frame
        if k != self.num_elements - 1 and self.directionality == "both":
            tv_loss += torch.sum(torch.abs(img - self.imgs[k + 1])) # TV to next frame

        if self.normalize == "average":
            tv_loss /= img.numel()
        return tv_loss

    def compute_and_update(self, k, img):
        tv_loss = self.compute(k, img)
        self.update_frame(k, img)

        return tv_loss


class ReconstructionMethod():
    def __init__(self, param):
        self.param = param

        self.fmlp = FMLP(**vars(param.fmlp)).type(dtype)
        
        parameters = [x for x in self.fmlp.parameters()]
        self.optimizer = torch.optim.Adam(parameters, **vars(self.param.optimizer))

        self.spatial_coordinate_grid = self.get_spatial_coordinate_grid().type(dtype)

        self.forward_operator = ForwardOperator().type(dtype)

        self.tvloss = TVLoss(**vars(param.tvloss))

    def get_spatial_coordinate_grid(self):
        """
        Generates a 2D-grid of (y, x) coordinates matching the Cartesian k-space sampling grid
        """
        fov_y = self.param.data.fov["y"]
        fov_x = self.param.data.fov["x"]
        voxel_size_y = fov_y / self.param.data.Ny
        voxel_size_x = fov_x / self.param.data.Nx

        y = torch.arange(start=voxel_size_y/2, end=fov_y, step=voxel_size_y)
        x = torch.arange(start=voxel_size_x/2, end=fov_x, step=voxel_size_x)

        coordinate_grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)

        return coordinate_grid

    def load_state(self, path, gpu):
        states = torch.load(path, map_location=torch.device('cuda', gpu))
        self.fmlp.load_state_dict(states["fmlp_state_dict"])
        self.optimizer.load_state_dict(states["optimizer"])

    def save_state(self, path):
        torch.save({
            'fmlp_state_dict': self.fmlp.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def reconstruction_loss(self, img, sample, smaps):
        kspace_hat = self.forward_operator(img, sample, smaps=smaps)

        squared_errors = torch.square(kspace_hat - sample["kspace"])
        squared_errors = torch.sum(torch.mean(squared_errors, dim=0)) # mean over batch, sum over rest

        return squared_errors

    def evaluate(self, sample):
        img = self.fmlp(self.spatial_coordinate_grid.flatten(end_dim=-2), sample["t_k"].type(dtype))
        img = img.reshape(1, self.param.data.Ny, self.param.data.Nx, self.param.fmlp.mlp_out_features)
        return img
    
    def evaluate_abs(self, sample):
        img = self.evaluate(sample)
        return root_sum_of_squares(img, dim=-1) # compute magnitude image

    def evaluate_npy(self, sample):
        return self.evaluate_abs(sample).detach().cpu().numpy().squeeze()

    def train(self, dataset, validation_dataset=None):
        # setup logging of the traing
        self.writer = SummaryWriter(log_dir=self.param.experiment.results_dir)

        # create a directory for models and intermediate images during training
        training_dir = os.path.join(self.param.experiment.results_dir, "training")
        create_dir(training_dir)
        
        # copy smaps to GPU and setup dataloader
        smaps = dataset.smaps.squeeze(dim=1).type(dtype)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collocate_concat)
        dataloader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collocate_concat)

        img = self.evaluate_npy(dataset[0])
        self.writer.add_image("initial img", img / np.max(img), 0, dataformats="HW")
        print("Starting training... result directory: {}".format(self.param.experiment.results_dir))
         
        # initialize stored images in tv loss
        if self.param.hp.lambda_tv > 0.:
            for sample in dataset:
                img = self.evaluate(sample)
                self.tvloss.update_frame(sample["indices"], img)

        max_ser = float('-inf')
        max_ser_epoch = 0
        max_ser_subset = float('-inf')

        # inital number of training epochs. It can be prolonged if self.param.hp.extend_training_until_no_new_ser_highscore is true.
        num_epochs = self.param.hp.num_iter

        i = 1
        while i <= num_epochs: # iterate over epochs
            start_time = time.time()
            
            loss_avg = 0.
            loss_reconstruction_avg = 0.
            loss_tv_avg = 0.
            num_samples = 0.

            for sample in dataloader: # iterate over frames
                sample = copySampleToGPU(sample)

                self.optimizer.zero_grad()
                
                img = self.evaluate(sample)

                loss_reconstruction = self.reconstruction_loss(img, sample, smaps)
                if self.param.hp.lambda_tv > 0:
                    loss_tv = self.tvloss.compute_and_update(sample["indices"], img)
                else:
                    loss_tv = torch.tensor(0.).type(dtype)
                loss = loss_reconstruction + self.param.hp.lambda_tv * loss_tv

                loss.backward()
                self.optimizer.step()

                loss_reconstruction_avg += loss_reconstruction.detach()
                loss_tv_avg += loss_tv.detach()
                loss_avg += loss.detach()
                num_samples += 1 # kld and reconstruction loss is already averaged over the batch

            loss_reconstruction_avg /= num_samples
            loss_avg /= num_samples
            loss_tv_avg /= num_samples
            
            # save model parameters
            if i%self.param.experiment.model_save_frequency == 0 or i==self.param.hp.num_iter:
                # save intermediate model parameters and optimizer state
                self.save_state(os.path.join(training_dir, "trained_model_{}.pth".format(i)))
            
            # save a reconstructed video to TensorBoard
            if i%self.param.experiment.video_evaluation_frequency == 0 or i==self.param.hp.num_iter-1 or i == 1:
                imgs = torch.stack([self.evaluate_abs(sample).detach().cpu() for sample in dataset], dim=0)
                if "max_intensity_value" in vars(self.param.data).keys():
                    imgs /= self.param.data.max_intensity_value
                else:
                    imgs /= torch.max(imgs)
                imgs = imgs.unsqueeze(dim=0) # format to: B=1, N, C=1, H, W
                self.writer.add_video("video", imgs, i, self.param.data.frame_rate)

            # compute validation metrics
            if validation_dataset is not None and i%self.param.experiment.validation_evaluation_frequency == 0 or i==self.param.hp.num_iter-1 or i == 1:
                ser, ser_subset = self.evaluate_validation(dataloader_validation)

                self.writer.add_scalar('performance/validation_ser', ser, i)
                self.writer.add_scalar('performance/validation_ser_subset', ser_subset, i)
                
                if max_ser < ser:
                    max_ser = ser
                    max_ser_epoch = i
                    self.save_state(os.path.join(training_dir, "ser_highscore.pth"))

                if max_ser_subset < ser_subset:
                    max_ser_subset = ser_subset
                    self.save_state(os.path.join(training_dir, "ser_subset_highscore.pth"))

                self.writer.add_scalar('performance/max_validation_ser', max_ser, i)
                self.writer.add_scalar('performance/max_validation_ser_subset', max_ser_subset, i)
                
                if self.param.hp.extend_training_until_no_new_ser_highscore:
                    num_epochs = max(num_epochs, max_ser_epoch + self.param.hp.num_epochs_after_last_highscore)
            
            stop_time = time.time()
            eta = (stop_time - start_time)*(num_epochs - (i-1))/60

            self.writer.add_scalar('train/loss', loss_avg, i)
            self.writer.add_scalar('train/loss_reconstruction', loss_reconstruction_avg, i)
            self.writer.add_scalar('train/loss_tv', loss_tv_avg, i)

            print("Iteration {}: Train loss {:.7f}, reconstruction: {:.7f}, tv: {:.7f}, eta: {:.1f}min".format(i, loss_avg, loss_reconstruction_avg, loss_tv_avg, eta), '\r', end='')
            
            i += 1
        print("\n") # clear stdout
    

    def evaluate_validation(self, dataloader_validation):
        smaps = dataloader_validation.dataset.smaps.squeeze(dim=1).type(dtype) # copy to GPU

        squared_error = torch.tensor(0., dtype=torch.float64)
        squared_signal = torch.tensor(0., dtype=torch.float64)
        squared_error_subset = torch.tensor(0., dtype=torch.float64)
        squared_signal_subset = torch.tensor(0., dtype=torch.float64)

        frame_times = self.param.data.frame_times.type(dtype)

        for sample in dataloader_validation:
            sample = copySampleToGPU(sample)
            with torch.no_grad():
                # find training frame that is closest in time
                k = torch.argmin(torch.abs(frame_times - sample["t_k"]))
                sample["t_k"] = frame_times[k]
                img = self.evaluate(sample)

            kspace_rec = self.forward_operator.forward(img, sample, smaps=smaps)
                
            se = torch.sum(torch.square(kspace_rec - sample["kspace"]).flatten(start_dim=1), dim=-1).detach() # squared error
            ss = torch.sum(torch.square(sample["kspace"]).flatten(start_dim=1), dim=-1).detach() # squared signal
            
            squared_error += torch.sum(se).cpu()
            squared_signal += torch.sum(ss).cpu()

            # check which of the elements in the batch are within the validation subset
            # sample["line_indices"]: (Ns, 1)
            is_in_subset = (sample["line_indices"].flatten() <= self.param.experiment.validation_subset_max_line_index)*1.

            squared_error_subset += torch.sum(is_in_subset * se).cpu()
            squared_signal_subset += torch.sum(is_in_subset * ss).cpu()

        ser = 10. * torch.log10(squared_signal / squared_error)
        ser_subset = 10. * torch.log10(squared_signal_subset / squared_error_subset)

        return ser, ser_subset
    

    def transform(self, sample):

        # compute the measurement time of the measured coordinates assuming that the time is constant within every measured line.
        _, Nl, Nr, _ = sample["trajectory"].shape
        t_coordinates = (self.param.data.tr * sample["line_indices"].unsqueeze(dim=-1)).repeat((1, 1, Nr)) # (Ns, Nl, Nr)

        # compute the time at the center of the frames
        t_k = self.param.data.tr / 2. * (sample["line_indices"][:, 0] + sample["line_indices"][:, -1]).type(torch.float32)

        new_sample = {
            "kspace": sample["kspace"],
            "t_k": t_k,
            "t_coordinates": t_coordinates,
            "line_indices": sample["line_indices"],
            "indices": sample["indices"],
            "mask": sample["mask"]
        }

        return new_sample