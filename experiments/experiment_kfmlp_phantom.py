import os, argparse
from types import SimpleNamespace

import torch
import random
import numpy as np

from src import *
from data.scanner.CAVA_V1 import datasets_cava_v1

from torch.utils.tensorboard import SummaryWriter


def load_dataset(param, model=None):
    transform = None
    if model is not None:
        transform = model.transform

    cartesian_dataset = CartesianDataset.from_npyfiles(param.data.base_path_and_name)
    dataset, validation_dataset = SparseCartesianDataset.from_cartesian_dataset_extract_validation_dataset_rebin(cartesian_dataset,
                                                                                                                 param.data.listfile_path,
                                                                                                                 validation_percentage=param.data.validation_percentage,
                                                                                                                 number_of_lines_per_frame=param.data.number_of_lines_per_frame,
                                                                                                                 max_Nk=param.data.Nk,
                                                                                                                 transform=transform)
    return dataset, validation_dataset

## main function that executes an experiment
def run_experiment(param):
    # reset seeds for each experiment
    np.random.seed(1998)
    random.seed(1998)
    torch.manual_seed(1998)

    # copy the model and the experiment script to the results directory
    create_dir(param.experiment.results_dir)
    os.popen('cp {} \'{}\''.format(param.experiment.model_file_path, param.experiment.results_dir))
    new_model_file_name = os.path.join(param.experiment.results_dir, os.path.basename(param.experiment.model_file_path))
    os.popen('cp {} \'{}\''.format(param.experiment.script_file_path, param.experiment.results_dir))

    models = import_file(new_model_file_name)
    model = models.ReconstructionMethod(param)

    ## Configure a cached subset of the dataset
    dataset, validation_dataset = load_dataset(param, model)                                                                                         
    dataset = DatasetCache(dataset.subset(param.data.sample_indices), max_numel_gpu=0)
    validation_dataset = DatasetCache(validation_dataset, max_numel_gpu=0)

    # save parameters
    torch.save(param, os.path.join(param.experiment.results_dir, "param.pth"))
    with open(os.path.join(param.experiment.results_dir, "param.txt"), "w") as text_file:
        text_file.write("{}".format(param))

    ## Training
    model.train(dataset, validation_dataset)

## main function that loads an experiment
def load_experiment(results_dir, model_param_file_name, gpu=1):

    param = torch.load(os.path.join(results_dir, "param.pth"))
    param.experiment.results_dir = results_dir

    model_file_name = os.path.join(results_dir, os.path.basename(param.experiment.model_file_path))
    models = import_file(model_file_name)
    model = models.ReconstructionMethod(param)

    dataset, validation_dataset = load_dataset(param, model)                                                                                               
    dataset = dataset.subset(param.data.sample_indices)
                           
    model.load_state(os.path.join(param.experiment.results_dir, model_param_file_name), gpu)

    experiment = SimpleNamespace()
    experiment.param = param
    experiment.model = model
    experiment.dataset = dataset
    experiment.validation_dataset = validation_dataset

    return experiment


## main function that launches experiments
if __name__ == '__main__':
    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Specify the GPU device number.')
    args = parser.parse_args()

    ## Setup
    # configure GPU
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    dtype = torch.cuda.FloatTensor
    torch.backends.cudnn.enabled = True

    print("Selected GPU", gpu)

    for u in [1]:
        for i in [2]:

            Nk = 225
            sigma = 0. # 1e1
            out_scale = 1000.
            st = 1.0
            sx = 1e1
            lambda_denoiser = 0. # 0.1
            epsilon = 0. # 1e4

            np.random.seed(1998)
            random.seed(1998)
            torch.manual_seed(1998)

            param = SimpleNamespace()
            param.experiment = SimpleNamespace()
            param.data  = SimpleNamespace()
            param.hp = SimpleNamespace()
            param.fmlp = SimpleNamespace()
            param.optimizer = SimpleNamespace()
            param.metrics = SimpleNamespace()

            ## Dataset Configuration
            param.data.listfile_path = "data/xcat/phantom3/low_res_as_cava_v1_10.list"
            param.data.base_path_and_name = "data/xcat/phantom3/low_res_as_cava_v1_10"
            param.data.dataset_type = "sparse_cartesian"
            param.data.number_of_lines_per_frame = 6
            param.data.validation_percentage = 5
            param.data.Nk = Nk
            param.data.sample_indices = list(range(param.data.Nk))
            param.data.max_intensity_value = 12. # use the brighness value that is stored in the dataset_info (this value was set arbitrarily to obtain images with reasonable contrast)
            param.data.smaps_zero_threshold = 1e-5

            param.data.tr = 2.8e-3
            param.data.frame_rate = 1 / (param.data.tr * param.data.number_of_lines_per_frame) # approximately if validation_percentage is low

            dataset, validation_dataset = load_dataset(param)
            dataset = dataset.subset(param.data.sample_indices)
            (Nk, Nc, _, Ny, Nx) = dataset.shape()
            param.data.Nx = Nx
            param.data.Ny = Ny
            param.data.Nc = Nc

            param.data.frame_times = param.data.tr * (dataset.line_indices[:, 0] + dataset.line_indices[:, -1]) / 2 # t_k

            # KFMLP parameters
            param.fmlp.spatial_feature_map = "fourier_features"
            param.fmlp.resolution = [param.data.Ny, param.data.Nx]

            param.fmlp.spatial_in_features = 2
            param.fmlp.spatial_fmap_width = 512
            param.fmlp.spatial_coordinate_scales = [sx, sx] # spatial coordinate scale
            
            param.fmlp.temporal_in_features = 1
            param.fmlp.temporal_fmap_width = 128
            param.fmlp.temporal_coordinate_scales = [st] # temporal coordinate scale in [1/s]

            param.fmlp.mlp_width = 512
            param.fmlp.mlp_sigma = 0.01
            param.fmlp.mlp_scale = 1.
            param.fmlp.mlp_hidden_layers = 7
            param.fmlp.mlp_hidden_bias = True

            param.fmlp.mlp_out_features = 2 * Nc
            param.fmlp.mlp_final_sigma = 0.01
            param.fmlp.mlp_final_bias = True

            param.fmlp.out_scale = out_scale
            
            ## optimizer parameters
            param.optimizer.weight_decay = 0
            param.optimizer.lr = 2e-4

            ## other hyperparameters
            param.hp.num_iter = 100
            param.hp.extend_training_until_no_new_ser_highscore = True
            param.hp.num_epochs_after_last_highscore = 200
            param.hp.epsilon = epsilon
            param.hp.sigma = sigma
            param.hp.lambda_denoising_loss = lambda_denoiser
            param.hp.loss_type = "l_2"
            param.hp.batch_size_training = 1
            param.hp.batch_size_validation = 1

            text_description = "s_t {} sx {} out_scale {} eps {} sigma {} lambda {} reference".format(st, param.fmlp.spatial_coordinate_scales[0], param.fmlp.out_scale, param.hp.epsilon, param.hp.sigma, param.hp.lambda_denoising_loss)
            
            ## Experiment configuration
            param_series = SimpleNamespace()
            param_series.series_dir = "results/phantom3/KFMLP/validation/{}/l_2/".format(param.data.Nk)
            create_dir(param_series.series_dir)

            # copy all additional files to the series directory (so they are not changed during execution)
            experiment_script_path = __file__
            main_model_path = "src/models/kspace-fmlp.py"
            os.popen('cp {} {}'.format(experiment_script_path, param_series.series_dir))
            os.popen('cp {} {}'.format(main_model_path, param_series.series_dir))
            series_script_path = os.path.join(param_series.series_dir, os.path.basename(experiment_script_path))
            series_model_path = os.path.join(param_series.series_dir, os.path.basename(main_model_path))

            ## basic parameters
            param.experiment.results_dir = os.path.join(param_series.series_dir, text_description)
            param.experiment.model_file_path = series_model_path
            param.experiment.script_file_path = series_script_path
            param.experiment.model_save_frequency = 100
            param.experiment.video_evaluation_frequency = 100
            param.experiment.validation_evaluation_frequency = 1
            param.experiment.evaluate_reference_metrics = True
            param.experiment.reference_evaluation_frequency = 1

            param.metrics.img_scaling = 1.
            param.metrics.ssim=True
            param.metrics.psnr=False
            param.metrics.ser=False
            param.metrics.hfen=False
            param.metrics.brisque=False
            param.metrics.vif=True
            param.metrics.mse=True
            param.metrics.crossection_vif=False

            param.experiment.validation_subset_max_line_index = torch.max(dataset[224]["line_indices"])

            # free memory
            del dataset, validation_dataset

            print("Running experiment", param.experiment.results_dir, "...")
            run_experiment(param)
        

