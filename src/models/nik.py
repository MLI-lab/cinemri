import os, time
import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.data import Dataset as TDataset
from torch.utils.data import DataLoader
from src import root_sum_of_squares, copySampleToGPU, copySampleToCPU, dtype, ForwardOperator, collocate_concat, create_dir

from torch.utils.tensorboard import SummaryWriter

# code for NIK is copied and adapted from: https://github.com/wenqihuang/NIK_MRI

def k2img(k, csm=None, im_size=None, norm_factor=1):
    """
    Convert k-space to image space
    :param k: k-space data on a Cartesian grid
    :param csm: coil sensitivity maps
    :return: image
    """

    coil_img = ifft2c_mri(k)
    if im_size is not None:
        assert coil_img.shape == im_size
        # coil_img = center_crop(coil_img, im_size)
        if csm is not None:
            assert csm.shape == im_size
            # csm = center_crop(csm, im_size)

    k_mag = k[:,4,:,:].abs().unsqueeze(1).detach().cpu().numpy()        # nt, nx, ny   
    # combined_img_motion = coil_img_motion.abs()
    if csm is not None:
        if len(csm.shape) == len(coil_img.shape):
            im_shape = csm.shape[2:]        # (nx, ny)
        else:
            im_shape = csm.shape[1:]        # (nx, ny)
        combined_img = coilcombine(coil_img, im_shape, coil_dim=1, csm=csm)
    else:
        combined_img = coilcombine(coil_img, coil_dim=1, mode='rss')
    combined_phase = torch.angle(combined_img).detach().cpu().numpy()
    combined_mag = combined_img.abs().detach().cpu().numpy()
    k_mag = np.log(np.abs(k_mag) + 1e-4)
    
    k_min = np.min(k_mag)
    k_max = np.max(k_mag)
    max_int = 255

    # combined_mag_nocenter = combined_mag
    # combined_mag_nocenter[:,:,combined_img.shape[-2]//2-10:combined_img.shape[-2]//2+10,combined_img.shape[-1]//2-10:combined_img.shape[-1]//2+10] = 0
    combined_mag_max = combined_mag.max() / norm_factor

    k_mag = (k_mag - k_min)*(max_int)/(k_max - k_min)
    k_mag = np.minimum(max_int, np.maximum(0.0, k_mag))
    k_mag = k_mag.astype(np.uint8)
    combined_mag = (combined_mag / combined_mag_max * 255)#.astype(np.uint8)
    combined_phase = angle2color(combined_phase, cmap='viridis', vmin=-np.pi, vmax=np.pi)
    k_mag = np.clip(k_mag, 0, 255).astype(np.uint8)
    combined_mag = np.clip(combined_mag, 0, 255).astype(np.uint8)
    combined_phase = np.clip(combined_phase, 0, 255).astype(np.uint8)

    combined_img = combined_img.detach().cpu().numpy()
    vis_dic = {
        'k_mag': k_mag, 
        'combined_mag': combined_mag, 
        'combined_phase': combined_phase, 
        'combined_img': combined_img
    }
    return vis_dic

def angle2color(value_arr, cmap='viridis', vmin=None, vmax=None):
    """
    Convert a value to a color using a colormap
    :param value: the value to convert
    :param cmap: the colormap to use
    :return: the color
    """
    if vmin is None:
        vmin = value_arr.min()
    if vmax is None:
        vmax = value_arr.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    try:
        value_arr = value_arr.squeeze(0)
    except:
        value_arr = value_arr.squeeze()
    if len(value_arr.shape) == 3:
        color_arr = np.zeros((*value_arr.shape, 4))
        for i in range(value_arr.shape[0]):
            color_arr[i] = mapper.to_rgba(value_arr[i], bytes=True)
        color_arr = color_arr.transpose(0, 3, 1, 2)
    elif len(value_arr.shape) == 2:
        color_arr = mapper.to_rgba(value_arr, bytes=True)
    return color_arr


def ifft2c_mri(k):
    x = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(k, (-2,-1)), norm='ortho'), (-2,-1))
    return x

def fft2c_mri(img):
    k = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img, (-2,-1)), norm='ortho'), (-2,-1))
    return k

def coilcombine(img, im_shape=None, coil_dim=-1, mode='csm', csm=None):
    if mode == 'rss':
        return torch.sqrt(torch.sum(img**2, dim=coil_dim, keepdim=True))
    elif mode == 'csm':
        # csm = csm.unsqueeze(0)
        csm = torch.from_numpy(csm).to(img.device)

        # print(img.shape, im_shape)
        # assert img.shape == im_shape
        # assert csm.shape == im_shape

        #img = center_crop(img, im_shape)
        #csm = center_crop(csm, im_shape)
        return torch.sum(img*torch.conj(csm), dim=coil_dim, keepdim=True)
    else:
        raise NotImplementedError

class HDRLoss_FF(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['hdr_eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, kcoords, weights=None, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()
        dist_to_center2 = kcoords[...,2]**2 + kcoords[...,3]**2
        filter_value = torch.exp(-dist_to_center2/(2*self.sigma**2)).unsqueeze(-1)

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target
        # error = error * filter_value

        loss = (error.abs()/(input.detach().abs()+self.eps))**2


        if weights is not None:
            loss = loss * weights.unsqueeze(-1)

        reg_error = (input - input * filter_value)
        reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        

        if reduce:
            return loss.mean() + reg.mean(), reg.mean()
        else:
            return loss, reg
        

class AdaptiveHDRLoss(torch.nn.Module):
    """
    HDR loss function with frequency filtering (v4)
    """
    def __init__(self, config):
        super().__init__()
        self.sigma = float(config['hdr_ff_sigma'])
        self.eps = float(config['eps'])
        self.factor = float(config['hdr_ff_factor'])

    def forward(self, input, target, reduce=True):
        # target_max = target.abs().max()
        # target /= target_max
        # input = input / target_max
        # input_nograd = input.clone()
        # input_nograd = input_nograd.detach()

        if input.dtype == torch.float:
            input = torch.view_as_complex(input) #* filter_value
        if target.dtype == torch.float:
            target = torch.view_as_complex(target)
        
        assert input.shape == target.shape
        error = input - target
        # error = error * filter_value

        loss = (-error.abs()/((input.detach().abs()+self.eps)**2))**2
        # if weights is not None:
        #     loss = loss * weights.unsqueeze(-1)

        # reg_error = (input - input * filter_value)
        # reg = self.factor * (reg_error.abs()/(input.detach().abs()+self.eps))**2
        # reg = torch.matmul(torch.conj(reg).t(), reg)
        # reg = reg.abs() * self.factor
        # reg = torch.zeros([1]).mean()

        if reduce:
            return loss.mean()


class RadialDataset(TDataset):
    def __init__(self, dataset, num_lines):
        super().__init__()
        
        # copy data from NonCartesianDataset dataset to the format used by NIK

        # attributes of NonCartesianDataset
        #self.kspace = kspace # shape (Nk, Nc, Nl, Nr, 2) torch.tensor float32
        #self.trajectory = trajectory # shape (Nk, Nl, Nr, 3) torch.tensor float32
        #self.mask = mask # shape (Nk, Nl, Nr, 3) LongTensor
        #self.line_indices = line_indices # shape (Nk, Nl) LongTensor
        #self.smaps = smaps # shape (Nc, Nz, Ny, Nx, 2) torch.tensor float32

        kdata = dataset.kspace.permute((1,0,2,3,4)).flatten(start_dim=1, end_dim=-3) # nc, nSpokes, nFE, 2
        kdata = torch.complex(kdata[..., 0], kdata[..., 1]) # convert to complex (nc, nSpokes, nFE)
        traj = dataset.trajectory.flatten(start_dim=0, end_dim=1)[:,:,1:] # nSpokes, nFE, 2

        # normalize k-space data
        self.kdata_normalization_constant = torch.max(torch.abs(kdata))
        kdata = kdata / self.kdata_normalization_constant

        # kdata: nc, nSpokes, nFE
        # traj: 2, nSpokes, nFE
        self.kdata_flat = torch.reshape(kdata, (-1, 1)) # nc*nSpokes*nFE, 1
        # self.kdata_org = self.data['kdata'].astype(np.complex64)
        nc, nSpokes, nFE = kdata.shape
        self.n_kpoints = self.kdata_flat.shape[0]
        
        # create coordinates from trajectory
        tspokes = dataset.line_indices.flatten() / (num_lines-1)  # nSpokes, normalize to 0...1

        kcoords = torch.zeros((nc, nSpokes, nFE, 4))
        kcoords[:,:,:,0] = torch.reshape(tspokes * 2 - 1, (1, nSpokes, 1)) # normalize to [-1, 1]
        kc = torch.linspace(-1, 1, nc)
        kcoords[:,:,:,1] = np.reshape(kc, [nc, 1, 1])                   # nc, 1, 1

        # traj has been normalized to [-pi, pi] -> normalize it to [-1, 1] # nSpokes, nFE, 2
        kcoords[:,:,:,2] = traj[:,:,0].unsqueeze(dim=0).tile((nc, 1, 1)) / torch.pi  # nc, nSpokes, nFE
        kcoords[:,:,:,3] = traj[:,:,1].unsqueeze(dim=0).tile((nc, 1, 1)) / torch.pi  # nc, nSpokes, nFE

        # torch.linspace(-1, 1-2/nx, nx)

        self.kcoords_flat = torch.reshape(kcoords, (-1, 4))
        # self.kdata_flat = self.kdata_flat.to(device)     # nc*nSpokes*nFE
        # self.kcoords_flat = self.kcoords_flat.to(device) # nc*nSpokes*nFE, 4

        csm = dataset.smaps.squeeze() # nc ny nx 2
        csm_rss = torch.sqrt(torch.sum(torch.sum(torch.square(csm), dim=0), dim=-1))
        csm_ss = torch.sum(torch.sum(torch.square(csm), dim=0), dim=-1)
        csm = torch.complex(csm[..., 0], csm[..., 1]) # nc ny nx
        csm = np.nan_to_num(csm/csm_ss)
        self.csm = csm


    def __len__(self):
        return self.n_kpoints

    def __getitem__(self, index):
        # point wise sampling
        sample = {
            'coords': self.kcoords_flat[index],
            'targets': self.kdata_flat[index]
        }
        return sample


class ReconstructionMethod():

    def __init__(self, param):
        self.param = param
        self.config = vars(self.param.nik)
        self.NIKmodel = NIKSiren(vars(param.nik))

        self.forward_operator = ForwardOperator().type(dtype)

    def load_state(self, path, gpu):
        states = torch.load(path, map_location=torch.device('cuda', gpu))
        self.NIKmodel.load_state_dict(states["nik_state_dict"])
        # self.NIKmodel.optimizer.load_state_dict(states["optimizer"])

    def save_state(self, path):
        torch.save({
            'nik_state_dict': self.NIKmodel.state_dict(),
            'optimizer': self.NIKmodel.optimizer.state_dict()
        }, path)

    def set_csm(self, csm):
        self.csm = csm

    def evaluate(self, tcoords):
        img = self.NIKmodel.evaluate_on_grid(tcoords, self.csm) # nt ny nx complex
        img = img.squeeze(dim=1) # squeeze coils
        img = img * self.param.data.kdata_normalization_constant
        img = torch.stack((torch.real(img), torch.imag(img)), dim=-1) # nt ny nx 2
        return img
         
    def evaluate_abs(self, tcoords):
        img = self.evaluate(tcoords)
        return root_sum_of_squares(img, dim=-1) # compute magnitude image or RSS image

    def evaluate_npy(self, tcoords):
        return self.evaluate_abs(tcoords).detach().cpu().numpy().squeeze()

    def evaluate_validation(self, dataloader_validation):
        smaps = dataloader_validation.dataset.smaps.squeeze(dim=1).type(dtype) # copy to GPU

        squared_error = torch.tensor(0., dtype=torch.float64)
        squared_signal = torch.tensor(0., dtype=torch.float64)
        squared_error_subset = torch.tensor(0., dtype=torch.float64)
        squared_signal_subset = torch.tensor(0., dtype=torch.float64)

        # frame_times = self.param.data.frame_times.type(dtype)

        for sample in dataloader_validation:
            # sample = copySampleToGPU(sample)
            with torch.no_grad():
                # find training frame that is closest in time
                # k = torch.argmin(torch.abs(frame_times - sample["t_k"]))
                # print("validate: {}".format(sample["tcoord"].cpu()))
                img = self.evaluate(sample["tcoord"])

            sample = copySampleToGPU(sample)

            kspace_rec = self.forward_operator.forward(img, sample, smaps)

            se = torch.sum(torch.square(kspace_rec - sample["kspace"]).flatten(start_dim=1), dim=-1).detach() # squared error
            ss = torch.sum(torch.square(sample["kspace"]).flatten(start_dim=1), dim=-1).detach() # squared signal
            
            squared_error += torch.sum(se).cpu()
            squared_signal += torch.sum(ss).cpu()

            # check which of the elements in the batch are within the validation subset
            # sample["line_indices"]: (Ns, 1)
            is_in_subset = (sample["line_indices"].flatten() <= self.param.experiment.validation_subset_max_line_index)*1.

            # sample = copySampleToCPU(sample)

            squared_error_subset += torch.sum(is_in_subset * se).cpu()
            squared_signal_subset += torch.sum(is_in_subset * ss).cpu()

        ser = 10. * torch.log10(squared_signal / squared_error)
        ser_subset = 10. * torch.log10(squared_signal_subset / squared_error_subset)

        return ser, ser_subset

    def train(self, dataset, validation_dataset=None):

        self.NIKmodel.network.train()
        self.NIKmodel.create_criterion()
        self.NIKmodel.create_optimizer()
        self.set_csm(dataset.csm)

        # setup logging of the traing
        self.writer = SummaryWriter(log_dir=self.param.experiment.results_dir)

        # create a directory for models and intermediate images during training
        training_dir = os.path.join(self.param.experiment.results_dir, "training")
        create_dir(training_dir)

        img = self.evaluate_npy(torch.tensor([0.]))
        self.writer.add_image("initial img", img / np.max(img), 0, dataformats="HW")
        print("Starting training... result directory: {}".format(self.param.experiment.results_dir))

        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        dataloader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=False, collate_fn=collocate_concat)
        
        max_ser = float('-inf')
        max_ser_epoch = 0
        max_ser_subset = float('-inf')

        # inital number of training epochs. It can be prolonged if self.param.hp.extend_training_until_no_new_ser_highscore is true.
        num_epochs = self.param.hp.num_iter

        accumulated_time = 0.

        i = 1
        while i <= num_epochs: # iterate over epochs
            start_time = time.time()

            loss_epoch = 0
            for b, sample in enumerate(dataloader):
                # kcoord, kv = sample['coords'], sample['target']
                loss = self.NIKmodel.train_batch(sample)
                # print(f"Epoch: {epoch}, Iter: {b}, Loss: {loss}")
                loss_epoch += loss

            print(f"Epoch: {i-1}, Epoch Loss: {loss_epoch}")

            stop_time = time.time()
            accumulated_time +=  stop_time - start_time
            self.writer.add_scalar('performance/training_time', accumulated_time, i)

            # save model parameters
            if i%self.param.experiment.model_save_frequency == 0 or i==self.param.hp.num_iter:
                # save intermediate model parameters and optimizer state
                self.save_state(os.path.join(training_dir, "trained_model_{}.pth".format(i)))
            
            # save a reconstructed video to TensorBoard
            if i%self.param.experiment.video_evaluation_frequency == 0 or i==self.param.hp.num_iter-1:
                imgs = torch.stack([self.evaluate_abs(torch.tensor([t])).detach().cpu() for t in torch.linspace(-1, 1, 100)], dim=0)
                if "max_intensity_value" in vars(self.param.data).keys():
                    imgs /= self.param.data.max_intensity_value
                else:
                    imgs /= torch.max(imgs)
                imgs = imgs.unsqueeze(dim=0) # format to: B=1, N, C=1, H, W
                self.writer.add_video("video", imgs, i, self.param.data.frame_rate)

            if i%self.param.experiment.img_evaluation_frequency == 0 or i==self.param.hp.num_iter-1:
                img = self.evaluate_npy(torch.tensor([-1.]))
                self.writer.add_image("first img", img / np.max(img), i, dataformats="HW")

                test_sample = dataset[(27) * 264:(28) * 264]
                test_sample = self.NIKmodel.pre_process(test_sample)
                kpred = self.NIKmodel.forward(test_sample)
                kpred = self.NIKmodel.post_process(kpred).squeeze()
                fig, ax = plt.subplots()
                ax.plot(torch.linspace(-1, 1.-2/264, 264), torch.abs(test_sample["targets"].squeeze()).cpu().detach().numpy())
                ax.plot(torch.linspace(-1, 1.-2/264, 264), torch.abs(kpred).cpu().detach().numpy())
                self.writer.add_figure("figure/kspace", fig, i)

            # compute validation metrics
            if validation_dataset is not None and (i%self.param.experiment.validation_evaluation_frequency == 0 or i==self.param.hp.num_iter-1 or i == 1):
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
            
            self.writer.add_scalar('train/loss', loss_epoch, i)

            i += 1

            # eta = (stop_time - start_time)*(num_epochs - (i-1))/60
            # print("Iteration {}: Train loss {:.7f}, eta: {:.1f}min".format(i, loss_epoch, eta), '\r', end='')
            

    def transform(self, sample):
        tcoord = 2 * sample["line_indices"] / (self.param.data.num_lines - 1) - 1.

        return {
            "kspace": sample["kspace"],
            "mask": sample["mask"],
            "trajectory": sample["trajectory"],
            "tcoord": tcoord,
            "smaps": sample["smaps"],
            "line_indices": sample["line_indices"]
        }
    

class NIKSiren(nn.Module, ABC):
    def __init__(self, config) -> None:

        super().__init__()

        self.config = config
        # needed for both training and testing
        # will be set in corresponding functions
        self.device = torch.device('cuda')
        self.network = None
        self.output = None

        # needed for training
        self.model_save_path = None
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None

        B = torch.randn((self.config['coord_dim'], self.config['feature_dim']//2), dtype=torch.float32) * self.config["B_scaling"]
        self.register_buffer('B', B)
        self.create_network()
        self.to(self.device)

    def create_optimizer(self):
        """Create the optimizer."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(self.config['lr']))


    def create_criterion(self):
        """Create the loss function."""
        self.criterion = HDRLoss_FF(self.config)
        # self.criterion = torch.nn.MSELoss()
        # self.criterion = AdaptiveHDRLoss(self.config)

    def create_network(self):
        feature_dim = self.config["feature_dim"]
        num_layers = self.config["num_layers"]
        out_dim = self.config["out_dim"]
        self.network = Siren(feature_dim, num_layers, out_dim).to(self.device)

    def pre_process(self, inputs):
        """
        Preprocess the input coordinates.
        """
        inputs['coords'] = inputs['coords'].to(self.device)
        if inputs.keys().__contains__('targets'):
            inputs['targets'] = inputs['targets'].to(self.device)
        features = torch.cat([torch.sin(inputs['coords'] @ self.B),
                              torch.cos(inputs['coords'] @ self.B)] , dim=-1)
        inputs['features'] = features
        return inputs
    
    def post_process(self, output):
        """
        Convert the real output to a complex-valued output.
        The first half of the output is the real part, and the second half is the imaginary part.
        """
        output = torch.complex(output[...,0:self.config["out_dim"]], output[...,self.config["out_dim"]:])
        return output

    def train_batch(self, sample):
        self.optimizer.zero_grad()
        sample = self.pre_process(sample)
        output = self.forward(sample)
        output = self.post_process(output)
        loss, reg = self.criterion(output, sample['targets'], sample['coords'])
        # loss = self.criterion(torch.view_as_real(output), torch.view_as_real(sample['targets']))

        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu()
    
    def evaluate_on_grid(self, tcoords, csm):
        """
        Evaluate the network on a cartesian grid for a given time coordinate
        """
        with torch.no_grad():
            nx = self.config['nx']
            ny = self.config['ny']
            nt = tcoords.numel()

            ts = tcoords.flatten()
            nc = len(self.config['coil_select'])
            kc = torch.linspace(-1, 1, nc)
            kxs = torch.linspace(-1, 1-2/nx, nx)
            kys = torch.linspace(-1, 1-2/ny, ny)
            

            # TODO: disgard the outside coordinates before prediction
            # why are these coordinates discarded?
            # - with the radial sampling pattern, only coordinates inside are sampled
            # - with the cartesian sampling pattern, no coordiantes should be discarded -> comment out

            grid_coords = torch.stack(torch.meshgrid(ts.cpu(), kc, kys, kxs, indexing='ij'), -1).to(self.device) # nt, nc, ny, nx, 4
            # dist_to_center = torch.sqrt(grid_coords[:,:,:,:,2]**2 + grid_coords[:,:,:,:,3]**2)

            

            # split t for memory saving
            t_split = 1
            t_split_num = np.ceil(nt / t_split).astype(int)

            # split c for memory saving
            c_split = 2 # number of bins
            c_split_num = np.ceil(nc / c_split).astype(int)


            kpred_list = []
            for t_batch in range(t_split_num):

                c_kpred_list = []
                c_index = 0
                for c_batch in range(c_split):
                    c_index_end = min(nc, c_index + c_split_num)
                    grid_coords_batch = grid_coords[t_batch*t_split:(t_batch+1)*t_split, c_index:c_index_end]
                    grid_coords_batch = grid_coords_batch.reshape(-1, 4).requires_grad_(False)
                    # get prediction
                    sample = {'coords': grid_coords_batch}
                    sample = self.pre_process(sample)
                    kpred = self.forward(sample)
                    kpred = self.post_process(kpred)
                    c_kpred_list.append(kpred)
                    c_index = c_index_end

                kpred = torch.concat(c_kpred_list, 0)

                kpred_list.append(kpred)
            kpred = torch.concat(kpred_list, 0)
            
            # TODO: clearning this part of code
            kpred = kpred.reshape(nt, nc, ny, nx)
            #k_outer = 1
            #kpred[dist_to_center>=k_outer] = 0
            coil_img = ifft2c_mri(kpred)
            combined_img = coilcombine(coil_img, coil_dim=1, csm=csm)

            return combined_img
    
    def forward(self, inputs):
        return self.network(inputs['features'])

"""
The following code is a demo of mlp with sine activation function.
We suggest to only use the mlp model class to do the very specific 
mlp task: takes a feature vector and outputs a vector. The encoding 
and post-process of the input coordinates and output should be done 
outside of the mlp model (e.g. in the prepocess and postprocess 
function in your NIK model class).
"""

class Siren(nn.Module):
    def __init__(self, hidden_features, num_layers, out_dim, omega_0=30, exp_out=True) -> None:
        super().__init__()

        self.net = [SineLayer(hidden_features, hidden_features, is_first=True, omega_0=omega_0)]
        for i in range(num_layers-1):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=omega_0))
        final_linear = nn.Linear(hidden_features, out_dim*2)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega_0, 
                                          np.sqrt(6 / hidden_features) / omega_0)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, features):
        return self.net(features)



class SineLayer(nn.Module):
    """Linear layer with sine activation. Adapted from Siren repo"""
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
