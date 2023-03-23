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
        # return tmp
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

class DirichletFeatureMap(nn.Module):
    def __init__(self, in_features, out_features, sigma, resolution):
        super().__init__()
        assert out_features % (2 * in_features) == 0

        self.num_freq = out_features // (2 * in_features)
        self.out_features = out_features

        self.resolution = nn.Parameter(torch.tensor(resolution, dtype=torch.float32)) # ((Nz,) Ny, Nx)
        self.resolution.requires_grad = False

        self.frequencies = nn.Parameter(torch.zeros(in_features, self.num_freq)) 
        self.frequencies.requires_grad = False

        with torch.no_grad():
            self.frequencies.normal_(std=1, mean=0)
            self.frequencies *= torch.tensor(sigma).squeeze().unsqueeze(dim=-1) # sigma: (in_features), std of the Gaussian frequency distribution in each dimension


    def forward(self, input):
        # input: (N, in_features)
        # output: (N, out_features) = (N, [ReDim1, ReDim2, ReDim3, ImDim1, ImDim2, ImDim3]), for in_features = 3

        # self.frequencies: (in_features, num_freq)

        n = torch.floor(self.resolution / 2)

        positive_shift = input.unsqueeze(dim=-1) + self.frequencies.unsqueeze(dim=0) # add (N, input_features, 1) and (1, input_features, num_freq) -> (N, input_features, num_freq)
        negative_shift = input.unsqueeze(dim=-1) - self.frequencies.unsqueeze(dim=0)

        positive_dirichlet = torch.sin((n + 0.5).unsqueeze(dim=0).unsqueeze(dim=-1) * positive_shift) / torch.sin(0.5 * positive_shift) # (N, input_features, num_freq)
        negative_dirichlet = torch.sin((n + 0.5).unsqueeze(dim=0).unsqueeze(dim=-1) * negative_shift) / torch.sin(0.5 * negative_shift) # (N, input_features, num_freq)

        cos = (0.5 / torch.sqrt(self.resolution)).unsqueeze(dim=0).unsqueeze(dim=-1) * (positive_dirichlet + negative_dirichlet) # (N, input_features, num_freq)
        sin = (0.5 / torch.sqrt(self.resolution)).unsqueeze(dim=0).unsqueeze(dim=-1) * (negative_dirichlet - positive_dirichlet) # (N, input_features, num_freq)

        correction = 2.*torch.cos(self.frequencies).unsqueeze(dim=0)*torch.cos(0.5*self.resolution.unsqueeze(dim=0)*input).unsqueeze(dim=-1) * (self.resolution % 2.).unsqueeze(dim=0).unsqueeze(dim=-1) # (N, input_features, num_freq)
        cos += correction

        return Normalization(torch.concat((
            cos.flatten(start_dim=1),
            sin.flatten(start_dim=1),
        ), dim=-1)) # (N, 2 * input_features * num_freq = output_features)
    

# class DirichletFeatureMap(nn.Module):
#     def __init__(self, in_features, out_features, sigma, resolution):
#         super().__init__()
#         assert out_features % (2 * in_features) == 0

#         self.num_freq = out_features // 2
#         self.out_features = out_features

#         self.resolution = nn.Parameter(torch.tensor(resolution, dtype=torch.float32)) # ((Nz,) Ny, Nx)
#         self.resolution.requires_grad = False

#         self.frequencies = nn.Parameter(torch.zeros(in_features, self.num_freq)) 
#         self.frequencies.requires_grad = False

#         with torch.no_grad():
#             self.frequencies.normal_(std=1, mean=0)
#             self.frequencies *= torch.tensor(sigma).squeeze().unsqueeze(dim=-1) # sigma: (in_features), std of the Gaussian frequency distribution in each dimension


#     def forward(self, input):
#         # input: (N, in_features)
#         # output: (N, out_features) = (N, [ReDim1, ReDim2, ReDim3, ImDim1, ImDim2, ImDim3]), for in_features = 3

#         # self.frequencies: (in_features, num_freq)

#         n = torch.floor(self.resolution / 2)

#         positive_shift = input.unsqueeze(dim=-1) + self.frequencies.unsqueeze(dim=0) # add (N, input_features, 1) and (1, input_features, num_freq) -> (N, input_features, num_freq)
#         negative_shift = input.unsqueeze(dim=-1) - self.frequencies.unsqueeze(dim=0)

#         positive_dirichlet = torch.sin((n + 0.5).unsqueeze(dim=0).unsqueeze(dim=-1) * positive_shift) / torch.sin(0.5 * positive_shift) # (N, input_features, num_freq)
#         negative_dirichlet = torch.sin((n + 0.5).unsqueeze(dim=0).unsqueeze(dim=-1) * negative_shift) / torch.sin(0.5 * negative_shift) # (N, input_features, num_freq)

#         pos_prod = torch.prod(positive_dirichlet, dim=1)
#         neg_prod = torch.prod(negative_dirichlet, dim=1)

#         cos = (0.5 / torch.prod(torch.sqrt(self.resolution))) * (pos_prod + neg_prod) # (N, num_freq)
#         sin = (0.5 / torch.prod(torch.sqrt(self.resolution))) * (neg_prod - pos_prod) # (N, num_freq)

#         # correction = 2.*torch.cos(self.frequencies).unsqueeze(dim=0)*torch.cos(0.5*self.resolution.unsqueeze(dim=0)*input).unsqueeze(dim=-1) * (self.resolution % 2.).unsqueeze(dim=0).unsqueeze(dim=-1) # (N, input_features, num_freq)
#         # cos += correction

#         return Normalization(torch.concat((
#             cos,
#             sin,
#         ), dim=-1)) # (N, 2 * num_freq = output_features)
        
# class DirichletFeatureMap(nn.Module):
#     def __init__(self, in_features, out_features, sigma, resolution):
#         super().__init__()
#         assert out_features % (2 * in_features) == 0

#         self.sigma = torch.tensor(sigma)

#         self.num_freq = out_features // 2
#         self.out_features = out_features

#         self.resolution = nn.Parameter(torch.tensor(resolution, dtype=torch.float32)) # ((Nz,) Ny, Nx)
#         self.resolution.requires_grad = False

#         self.linear = nn.Linear(in_features, self.num_freq, bias=False)
#         self.linear.weight.requires_grad = False
#         with torch.no_grad():
#             self.linear.weight.normal_(std=sigma[0], mean=0)

#         y = 2. * torch.pi * (-int(resolution[0]/2.) + torch.arange(float(resolution[0]))) / resolution[0]
#         x = 2. * torch.pi * (-int(resolution[1]/2.) + torch.arange(float(resolution[1]))) / resolution[1]
#         coordinate_grid = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=-1)
#         self.coordinate_grid = coordinate_grid
#         fmap_cos = np.sqrt(2)*torch.cos(self.linear(self.sigma*coordinate_grid) + torch.pi/4)
#         fmap_sin = np.sqrt(2)*torch.sin(self.linear(self.sigma*coordinate_grid) + torch.pi/4)
#         self.dfmap_cos = fft2(torch.stack((fmap_cos, torch.zeros_like(fmap_cos)), dim=-1))[..., 0] # only keep real part
#         self.dfmap_sin = fft2(torch.stack((fmap_sin, torch.zeros_like(fmap_cos)), dim=-1))[..., 1] # only keep imaginary part
        
#         fmap_ifft = ifft2(torch.stack((self.dfmap_cos, torch.zeros_like(fmap_cos)), dim=-1))[...,0]
#         print(torch.max(torch.abs(self.dfmap_cos[...,0]))), print(torch.max(torch.abs(self.dfmap_cos[...,1])))
#         plt.imshow(fmap_ifft[...,1])
        
#     def forward(self, input):
#         # input: (N, in_features)
#         # output: (N, out_features) = (N, [ReDim1, ReDim2, ReDim3, ImDim1, ImDim2, ImDim3]), for in_features = 3
#         input = self.coordinate_grid.flatten(end_dim=-2).type(dtype)
        
#         # self.frequencies: (in_features, num_freq)
#         n = torch.floor(self.resolution / 2)
#         indices = torch.round(self.resolution.unsqueeze(dim=0) * input / (2*torch.pi)) + n.unsqueeze(dim=0)
#         indices = indices.type(torch.int64)

#         return Normalization(torch.concat((
#             self.dfmap_cos[indices[:,0], indices[:,1], :],
#             self.dfmap_sin[indices[:,0], indices[:,1], :]
#         ), dim=-1)) # (N, 2 * num_freq = output_features)
        
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
                spatial_feature_map,

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

                out_scale,

                # extra parameters of the DirichletFeatureMap
                resolution=None
                ):
        super().__init__()

        self.spatial_in_features = spatial_in_features
        self.temporal_in_features = temporal_in_features

        if spatial_feature_map == "fourier_features":
            self.spatial_fmap = FourierFeatureMap(spatial_in_features, spatial_fmap_width, spatial_coordinate_scales)
        elif spatial_feature_map == "dirichlet_features":
            self.spatial_fmap = DirichletFeatureMap(spatial_in_features, spatial_fmap_width, spatial_coordinate_scales, resolution)

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
        else: # temporal coordinate is provided separately
            spatial_ff = self.spatial_fmap(coords)
            temporal_ff = self.temporal_fmap(temporal_coord)
            combined = torch.concat((spatial_ff, temporal_ff), dim=-1)
    
        return self.mlp(combined) * self.out_scale


class ReconstructionMethod():
    def __init__(self, param):
        self.param = param

        self.fmlp = FMLP(**vars(param.fmlp)).type(dtype)
        
        parameters = [x for x in self.fmlp.parameters()]
        self.optimizer = torch.optim.Adam(parameters, **vars(self.param.optimizer))

        self.kspace_coordinate_grid = self.get_kspace_coordinate_grid().type(dtype)

        self.forward_operator = ForwardOperator(dataset_type=self.param.data.dataset_type, Ny=param.data.Ny, Nx=param.data.Nx).type(dtype)

        self.weighted_smaps = None


    def get_kspace_coordinate_grid(self):
        """
        Generates a 2D-grid of (y, x) coordinates matching the Cartesian k-space sampling grid
        """
        Ny, Nx = self.param.data.Ny, self.param.data.Nx

        z = torch.tensor([0.])
        y = 2. * torch.pi * (-int(Ny/2.) + torch.arange(float(Ny))) / Ny
        x = 2. * torch.pi * (-int(Nx/2.) + torch.arange(float(Nx))) / Nx

        coordinate_grid = torch.stack(torch.meshgrid(z, y, x, indexing="ij"), dim=-1)

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

    def denoising_loss(self, kspace_hat, trajectory, sigma, epsilon):
        # kspace_hat: (1, Nc, Nl, Nr, 2)
        # trajectory: (1, Nl, Nr, 3)
        # sigma: strength of the denoising
        # epsilon: high dynamic range loss parameter
        distance = root_sum_of_squares(trajectory, dim=-1).unsqueeze(dim=-1).unsqueeze(dim=0) # 1 1 Nl Nr 1
        denoised = kspace_hat * torch.exp(-distance*(1/(2*(sigma**2))))
        squared_errors = torch.square((kspace_hat - denoised) / (torch.abs(kspace_hat.detach()) + epsilon))
        return torch.sum(torch.mean(squared_errors, dim=0)) # mean over batch, sum over rest

    def high_dynamic_range_loss(self, kspace_hat, sample, epsilon):
        squared_errors = torch.square((kspace_hat - sample["kspace"]) / (torch.abs(kspace_hat.detach()) + epsilon)) # stop the gradient for the weighting factor
        return torch.sum(torch.mean(squared_errors, dim=0)) # mean over batch, sum over rest

    def reconstruction_loss(self, kspace_hat, sample):
        squared_errors = torch.square(kspace_hat - sample["kspace"])
        return torch.sum(torch.mean(squared_errors, dim=0)) # mean over batch, sum over rest
    
    def evaluate_trajectory(self, trajectory, t_k=None, t_coordinates=None):
        # trajectory: (Ns, Nl, Nr, 3), 3: z y x
        # t_k: (Ns)
        # t_coordinates: (Ns, Nl, Nr)
        # output: (Ns, Nc, Nl, Nr, 2)
        # ! only implemented for 2D trajectories, yet (z coordinate is discarded)

        Ns, Nl, Nr, _ = trajectory.shape
        trajectory_flattened = trajectory.flatten(end_dim=-2) # (Ns*Nl*Nr, 3)

        if t_k is not None: # evaluate at a single time coordinate
            t_coordinates_flattened = t_k.reshape((Ns, 1, 1)).repeat((1, Nl, Nr)).flatten().unsqueeze(dim=-1) # (Ns*Nl*Nr, 1)
        elif t_coordinates is not None: # evaluate at a individual temporal coordinates
            t_coordinates_flattened = t_coordinates.flatten().unsqueeze(dim=-1) # (Ns*Nl*Nr, 1)
        else:
            raise Exception
        
        kspace_hat_flattened = self.fmlp(trajectory_flattened[:, 1:3], t_coordinates_flattened)  # (Ns*Nl*Nr, 2*Nc)
        return kspace_hat_flattened.reshape((Ns, Nl, Nr, -1, 2)).permute((0,3,1,2,4)) # (Ns, Nc, Nl, Nr, 2)
    
    def compute_weighted_smaps(self, smaps, eps=1e-3):
        # required smaps shape: (Nc, Ny, Nx, 2)
        magnitude_squared = torch.sum(torch.sum(torch.square(smaps), dim=-1, keepdim=True), dim=0, keepdim=True)
        magnitude_squared[magnitude_squared < eps] = torch.inf
        self.weighted_smaps = complex_conj(smaps) / magnitude_squared

    def combine_coils(self, kspace):
        assert self.weighted_smaps is not None

        img_coils = ifft2(kspace) # (Ns, Nc, Ny, Nx, 2)

        # perform coil combination: x = \sum_{c=1}^C x_c \odot s_c^*
        # smaps shape: (Nc, Ny, Nx, 2)
        img = torch.sum(complex_mul(self.weighted_smaps.unsqueeze(dim=0), img_coils), dim=-4)

        return img # Ns, Ny, Nx, 2

    def evaluate(self, sample):
        Ns = sample["t_k"].shape[0]
        trajectory = self.kspace_coordinate_grid.reshape((1, self.param.data.Ny, self.param.data.Nx, 3)).repeat((Ns, 1, 1, 1))
        kspace_hat = self.evaluate_trajectory(trajectory, t_k=sample["t_k"]) # (Ns, Nc, Ny, Nx, 2)
        img = self.combine_coils(kspace_hat)

        return img
    
    def evaluate_abs(self, sample):
        img = self.evaluate(sample)
        return root_sum_of_squares(img, dim=-1) # compute magnitude image

    def evaluate_npy(self, sample):
        with torch.no_grad():
            return self.evaluate_abs(sample).detach().cpu().numpy().squeeze()

    def train(self, dataset, validation_dataset=None):
        # setup logging of the traing
        self.writer = SummaryWriter(log_dir=self.param.experiment.results_dir)

        # create a directory for models and intermediate images during training
        training_dir = os.path.join(self.param.experiment.results_dir, "training")
        create_dir(training_dir)
        
        # copy smaps to GPU and setup dataloader
        smaps = dataset.smaps.type(dtype).squeeze()
        self.compute_weighted_smaps(smaps, eps=self.param.data.smaps_zero_threshold)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.param.hp.batch_size_training, shuffle=True, collate_fn=collocate_concat)
        dataloader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=self.param.hp.batch_size_validation, shuffle=False, collate_fn=collocate_concat)

        # img = self.evaluate_npy(copySampleToGPU(dataset[0]))
        # self.writer.add_image("initial img", img / np.max(img), 0, dataformats="HW")

        print("Starting training... result directory: {}".format(self.param.experiment.results_dir))

        max_ser = float('-inf')
        max_ser_epoch = 0
        max_ser_subset = float('-inf')

        max_ssim = 0.
        max_vif = 0.

        # inital number of training epochs. It can be prolonged if self.param.hp.extend_training_until_no_new_ser_highscore is true.
        num_epochs = self.param.hp.num_iter

        i = 1
        while i <= num_epochs: # iterate over epochs
            start_time = time.time()
            
            loss_avg = 0.
            loss_reconstruction_avg = 0.
            denoising_loss_avg = 0.
            num_samples = 0.

            for sample in dataloader: # iterate over frames
                sample = copySampleToGPU(sample)
                Ns = sample["kspace"].shape[0]

                self.optimizer.zero_grad()

                kspace_hat = self.evaluate_trajectory(sample["trajectory"], t_coordinates=sample["t_coordinates"])

                if self.param.hp.loss_type == "l_2":
                    loss_reconstruction = self.reconstruction_loss(kspace_hat, sample)
                elif self.param.hp.loss_type == "high_dynamic_range":
                    loss_reconstruction = self.high_dynamic_range_loss(kspace_hat, sample, self.param.hp.epsilon)
                else:
                    raise Exception

                denoising_loss = torch.tensor(0.).type(dtype)
                if self.param.hp.lambda_denoising_loss > 0.:
                    denoising_loss = self.denoising_loss(kspace_hat, sample["trajectory"], sigma=self.param.hp.sigma, epsilon=self.param.hp.epsilon)

                loss = loss_reconstruction + self.param.hp.lambda_denoising_loss * denoising_loss

                loss.backward()
                self.optimizer.step()

                loss_reconstruction_avg += loss_reconstruction.detach() * Ns
                denoising_loss_avg += denoising_loss.detach() * Ns
                loss_avg += loss.detach() * Ns
                num_samples += Ns

            loss_reconstruction_avg /= num_samples
            loss_avg /= num_samples
            denoising_loss_avg /= num_samples
            
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

            # compute full-reference image quality metrics
            if dataset.reference is not None and self.param.experiment.evaluate_reference_metrics and (i%self.param.experiment.reference_evaluation_frequency == 0 or i==self.param.hp.num_iter-1 or i == 1):
                self.metrics.clear()
                for sample in dataset:
                    img = self.evaluate_abs(sample).detach()
                    self.metrics.add(img.squeeze(), torch.tensor(dataset.reference[sample["indices"][0], 0, :, :]))
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
            self.writer.add_scalar('train/loss_denoising', denoising_loss_avg, i)

            print("Iteration {}: Train loss {:.7f}, reconstruction: {:.7f}, denoising: {:.7f}, eta: {:.1f}min".format(i, loss_avg, loss_reconstruction_avg, denoising_loss_avg, eta), '\r', end='')
            
            i += 1
        print("\n") # clear stdout
    

    def evaluate_validation(self, dataloader_validation):
        smaps = dataloader_validation.dataset.smaps.squeeze(dim=1).type(dtype) # copy to GPU

        squared_error = torch.tensor(0., dtype=torch.float64)
        squared_signal = torch.tensor(0., dtype=torch.float64)
        squared_error_subset = torch.tensor(0., dtype=torch.float64)
        squared_signal_subset = torch.tensor(0., dtype=torch.float64)

        for sample in dataloader_validation:
            sample = copySampleToGPU(sample)
            with torch.no_grad():
                img = self.evaluate(sample)

            if "mask" in sample.keys(): # cartesian dataset
                kspace_rec = self.forward_operator.forward_sparse_cartesian(img, smaps, sample["mask"])
            else:
                kspace_rec = self.forward_operator.forward_non_cartesian(img, smaps, sample["trajectory"])
                kspace_rec *= self.param.data.nufft_scaling_factor # compensate for scaling due to the NUFFT and the reconstruction via IFFT

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
            "trajectory": sample["trajectory"],
            "t_k": t_k,
            "t_coordinates": t_coordinates,
            "line_indices": sample["line_indices"],
            "indices": sample["indices"]
        }
        
        if "mask" in sample.keys():
            new_sample["mask"] = sample["mask"] # used for validation -> no rounding errors in indices

        return new_sample