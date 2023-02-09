import os, sys, argparse
from types import SimpleNamespace
import importlib.util

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

    if param.data.dataset_info["cartesian"]:
        dataset, validation_dataset = SparseCartesianDataset.from_sparse_matfile2d_extract_validation_dataset_rebin(param.data.dataset_info["matfile_path"],
                                                                                                                    param.data.dataset_info["listfile_path"],
                                                                                                                    remove_padding=param.data.remove_padding,
                                                                                                                    shift=param.data.shift,
                                                                                                                    set_smaps_outside_to_one=param.data.set_smaps_outside_to_one,
                                                                                                                    validation_percentage=param.data.validation_percentage,
                                                                                                                    number_of_lines_per_frame=param.data.number_of_lines_per_frame,
                                                                                                                    max_Nk=param.data.Nk,
                                                                                                                    transform=transform)
    else:
        dataset, validation_dataset = NonCartesianDataset3D.from_mat_file_binned_validation(param.data.dataset_info["matfile_path"],
                                                                                            param.data.dataset_info["listfile_path"],
                                                                                            param.data.number_of_lines_per_frame,
                                                                                            param.data.validation_percentage,
                                                                                            convert_to_rad=True,
                                                                                            has_norm_fac=True,
                                                                                            transpose_smaps=False,
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

    # import the model file
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(new_model_file_name))
    models = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = models
    spec.loader.exec_module(models)

    # instantiate the Model
    model = models.ReconstructionMethod(param)

    ## Configure a cached subset of the dataset
    dataset, validation_dataset = load_dataset(param, model)                                                                                         
    dataset = DatasetCache(dataset.subset(param.data.sample_indices), max_numel_gpu=1000)

    # save parameters
    torch.save(param, os.path.join(param.experiment.results_dir, "param.pth"))
    with open(os.path.join(param.experiment.results_dir, "param.txt"), "w") as text_file:
        text_file.write("{}".format(param))

    ## Training
    model.train(dataset, validation_dataset)

## main function that loads an experiment
def load_experiment(results_dir, model_param_file_name, gpu=1):

    experiment = SimpleNamespace()

    param = torch.load(os.path.join(results_dir, "param.pth"))
    param.experiment.results_dir = results_dir

    # copy the VAE model to the resuls directory for reproducability
    model_file_name = os.path.join(results_dir, os.path.basename(param.experiment.model_file_path))
    # import the vaemodel.py file
    spec = importlib.util.spec_from_file_location("module.name", os.path.abspath(model_file_name))
    models = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = models
    spec.loader.exec_module(models)

    # instantiate the VAE
    model = models.MultiResFMLP(param)

    dataset, validation_dataset = load_dataset(param, model)                                                                                               
    dataset = dataset.subset(param.data.sample_indices)
                           
    model.load_state(os.path.join(param.experiment.results_dir, model_param_file_name), gpu)

    experiment.param = param
    experiment.model = model
    experiment.dataset = dataset
    experiment.validation_set = validation_dataset

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

    for st, in zip([1.0]):

        np.random.seed(1998)
        random.seed(1998)
        torch.manual_seed(1998)

        cava_v1_measurement_number = 13
        dataset_info = datasets_cava_v1[cava_v1_measurement_number]

        param = SimpleNamespace()
        param.experiment = SimpleNamespace()
        param.data  = SimpleNamespace()
        param.hp = SimpleNamespace()
        param.fmlp = SimpleNamespace()
        param.optimizer = SimpleNamespace()
        param.metrics = SimpleNamespace()
        param.tvloss = SimpleNamespace()

        ## Dataset Configuration
        param.data.dataset_info = dataset_info
        param.data.remove_padding = True
        param.data.shift = True
        param.data.set_smaps_outside_to_one = True
        param.data.is_cartesian = dataset_info["cartesian"]
        if param.data.is_cartesian:
            param.data.dataset_type = "sparse_cartesian"
        else:
            param.data.dataset_type = "non_cartesian"
        param.data.number_of_lines_per_frame = 6
        param.data.validation_percentage = 5
        param.data.Nk = 225
        param.data.sample_indices = list(range(param.data.Nk))

        examcard = ExamCard(dataset_info["examcard_path"])
        param.data.tr = examcard.parameters["Act. TR/TE (ms)"][0] * 1e-3
        param.data.frame_rate = 1 / (param.data.tr * param.data.number_of_lines_per_frame) # approximately if validation_percentage is low

        # read shape of the dataset and compute noise variance

        dataset, validation_dataset = load_dataset(param)
        dataset = dataset.subset(param.data.sample_indices)
        (Nk, Nc, _, Ny, Nx) = dataset.shape()
        param.data.Nx = Nx
        param.data.Ny = Ny
        param.data.Nc = Nc

        param.data.frame_times = param.data.tr * (dataset.line_indices[:, 0] + dataset.line_indices[:, -1]) / 2

        # load Physlog and extract cardiac phase and respiratory state
        physlog = PhysLogData(dataset_info["physlog_path"])
        phys_info = physlog.single_marker_cardiac_and_respiratoy_info(sample_times=param.data.frame_times)
        param.data.cardiac_phases = phys_info["cardiac_phases"]
        param.data.cardiac_cycles = phys_info["cardiac_cycles"]

        param.data.fov = { # FOV [m]
            "y": 0.6,
            "x": 0.6
        } 

        # FMLP encoder parameters
        param.fmlp.spatial_in_features = 2
        param.fmlp.spatial_fmap_width = 512
        param.fmlp.spatial_coordinate_scales = [30., 30.] # spatial coordinate scale in [1/m]
                    
        param.fmlp.temporal_in_features = 1
        param.fmlp.temporal_fmap_width = 128
        param.fmlp.temporal_coordinate_scales = [st] # temporal coordinate scale in [1/s]

        param.fmlp.mlp_width = 512
        param.fmlp.mlp_sigma = 0.01
        param.fmlp.mlp_scale = 1.
        param.fmlp.mlp_hidden_layers = 7
        param.fmlp.mlp_hidden_bias = True

        param.fmlp.mlp_out_features = 2
        param.fmlp.mlp_final_sigma = 0.01
        param.fmlp.mlp_final_bias = True

        param.fmlp.out_scale = 120.
        
        ## optimizer parameters -> directly fed as arguments into the optimizer
        param.optimizer.weight_decay = 0
        param.optimizer.lr = 2e-4

        param.tvloss.num_elements = param.data.Nk
        param.tvloss.mode = "real_imag"
        param.tvloss.directionality = "both"
        param.tvloss.normalize = "false"

        ## other hyperparameters
        param.hp.num_iter = 100
        param.hp.extend_training_until_no_new_ser_highscore = True
        param.hp.num_epochs_after_last_highscore = 200
        param.hp.use_smaps = True
        param.hp.lambda_tv = 0.

        text_description = "paper_implementation, s_t {} spatial_coordinate_scale {}".format(st, param.fmlp.spatial_coordinate_scales[0])
        
        ## Experiment configuration
        param_series = SimpleNamespace()
        param_series.series_dir = "results/cava_v1/{}/FMLP/validation/{}/".format(cava_v1_measurement_number, param.data.Nk)
        create_dir(param_series.series_dir)

        # copy all additional files to the series directory (so they are not changed during execution)
        experiment_script_path = __file__
        main_model_path = "src/models/fmlp.py"
        os.popen('cp {} {}'.format(experiment_script_path, param_series.series_dir))
        os.popen('cp {} {}'.format(main_model_path, param_series.series_dir))
        series_script_path = os.path.join(param_series.series_dir, os.path.basename(experiment_script_path))
        series_model_path = os.path.join(param_series.series_dir, os.path.basename(main_model_path))

        ## Basic parameters
        param.experiment.results_dir = os.path.join(param_series.series_dir, text_description)
        param.experiment.model_file_path = series_model_path
        param.experiment.script_file_path = series_script_path
        param.experiment.model_save_frequency = 100
        param.experiment.video_evaluation_frequency = 20
        param.experiment.validation_evaluation_frequency = 1
        
        param.experiment.validation_subset_indices = []
        for v in validation_dataset:
            if v["k"] < 225:
                param.experiment.validation_subset_indices.append(v["line_index"])

        # free memory
        del dataset, validation_dataset

        print("Running experiment", param.experiment.results_dir, "...")
        run_experiment(param)
        

