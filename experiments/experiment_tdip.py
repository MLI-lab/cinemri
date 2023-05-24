import os, argparse
from types import SimpleNamespace

import torch
import random
import numpy as np

from src import *
from data import datasets

from torch.utils.tensorboard import SummaryWriter


def load_dataset(param, model=None):
    transform = None
    if model is not None:
        transform = model.transform
                                                     
    dataset, validation_dataset = Dataset.load_from_matfile(param.data.dataset_info["matfile_path"],
                                                            remove_padding=param.data.remove_padding,
                                                            set_smaps_outside_to_one=param.data.set_smaps_outside_to_one,
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


    for Nk in [225, 450, 900]:

        np.random.seed(1998)
        random.seed(1998)
        torch.manual_seed(1998)

        measurement_name = "lowres_highsnr"
        dataset_info = datasets[measurement_name]

        param = SimpleNamespace()
        param.experiment = SimpleNamespace()
        param.data  = SimpleNamespace()
        param.hp = SimpleNamespace()
        param.decoder = SimpleNamespace()
        param.optimizer = SimpleNamespace()
        param.metrics = SimpleNamespace()
        param.trajectory = SimpleNamespace()

        ## Dataset Configuration
        param.data.dataset_info = dataset_info
        param.data.remove_padding = True
        param.data.set_smaps_outside_to_one = True
        param.data.number_of_lines_per_frame = 6
        param.data.validation_percentage = 5
        param.data.Nk = Nk
        param.data.sample_indices = list(range(param.data.Nk))

        param.data.tr = dataset_info["tr"]
        param.data.frame_rate = 1 / (param.data.tr * param.data.number_of_lines_per_frame) # approximately if validation_percentage is low

        dataset, validation_dataset = load_dataset(param)
        dataset = dataset.subset(param.data.sample_indices)
        (Nk, Nc, _, Ny, Nx) = dataset.shape()
        param.data.Nx = Nx
        param.data.Ny = Ny
        param.data.Nc = Nc

        param.data.frame_times = param.data.tr * (dataset.line_indices[:, 0] + dataset.line_indices[:, -1]) / 2 # t_k

        # load cardiac phases
        mat = scipy.io.loadmat(dataset_info["matfile_path"])
        param.data.cardiac_phases = mat["ecg_cardiac_phases"][0][dataset.line_indices[:,0]]
        param.data.cardiac_cycles = mat["ecg_cardiac_cycles"][0][dataset.line_indices[:,0]]

        ## Decoder parameters
        param.decoder.in_features = 3
        param.decoder.out_features = 2
        param.decoder.out_size = [param.data.Ny, param.data.Nx]
        param.decoder.map_net_out_size = [6, 9]
        param.decoder.num_stages = 5
        param.decoder.num_conv = 2
        param.decoder.conv_channels = 256
        param.decoder.conv_bias = False
        param.decoder.output_scaling = 32.

        param.trajectory.type = "helix"
        param.trajectory.L = 3
        param.trajectory.z_slack = [0.5]
        param.trajectory.p = param.data.cardiac_cycles[param.data.Nk-1] + param.data.cardiac_phases[param.data.Nk-1] - param.data.cardiac_phases[0]
        param.trajectory.equal_frame_size = False

        print("cardiac cycles: ", param.data.cardiac_cycles[0], param.data.cardiac_cycles[param.data.Nk-1])
        print("cardiac phases: ", param.data.cardiac_phases[0], param.data.cardiac_phases[param.data.Nk-1])
        print("trajectory.p: ", param.trajectory.p)

        ## optimizer parameter
        param.optimizer.weight_decay = 0
        param.optimizer.lr = 1e-4

        ## other hyperparameters
        param.hp.num_iter = 100
        param.hp.extend_training_until_no_new_ser_highscore = True
        param.hp.num_epochs_after_last_highscore = 200
        param.hp.lambda_tv = 0.

        text_description = "z {} channels {}".format(param.trajectory.z_slack[0], param.decoder.conv_channels)
        
        ## Experiment configuration
        param_series = SimpleNamespace()
        param_series.series_dir = "results/{}/TDIP/validation/{}/".format(measurement_name, param.data.Nk)
        create_dir(param_series.series_dir)

        # copy all additional files to the series directory (so they are not changed during execution)
        experiment_script_path = __file__
        main_model_path = "src/models/tdip.py"
        os.popen('cp {} {}'.format(experiment_script_path, param_series.series_dir))
        os.popen('cp {} {}'.format(main_model_path, param_series.series_dir))
        series_script_path = os.path.join(param_series.series_dir, os.path.basename(experiment_script_path))
        series_model_path = os.path.join(param_series.series_dir, os.path.basename(main_model_path))

        ## basic parameters
        param.experiment.results_dir = os.path.join(param_series.series_dir, text_description)
        param.experiment.model_file_path = series_model_path
        param.experiment.script_file_path = series_script_path
        param.experiment.model_save_frequency = 100
        param.experiment.evaluation_frequency = 100
        param.experiment.video_evaluation_frequency = 100
        param.experiment.validation_evaluation_frequency = 1
        
        param.experiment.validation_subset_max_line_index = torch.max(dataset[224]["line_indices"])

        # free memory
        del dataset, validation_dataset

        print("Running experiment", param.experiment.results_dir, "...")
        run_experiment(param)
        

