# Useful commands

### Monitoring GPU usage
```bash
gpustat -cp --watch
```
Installation:
```bash
pip install gpustat
```

```bash
gpu-blame
```

### Starting Tensorboard

```bash
tensorboard --samples_per_plugin images=100 --logdir_spec tdip:results/cava_v1/10/TimedependentDIP,multires:results/cava_v1/10/MultiResFMLP

tensorboard --samples_per_plugin images=100 --logdir_spec tdip:results/cava_v1/13/TimedependentDIP,multires:results/cava_v1/13/MultiResFMLP

tensorboard --samples_per_plugin images=100 --logdir_spec tdip:results/cava_v1/15/TimedependentDIP,multires:results/cava_v1/15/MultiResFMLP

tensorboard --samples_per_plugin images=100 --logdir_spec tdip:results/cava_v1/20/TimedependentDIP,multires:results/cava_v1/20/MultiResFMLP

tensorboard --samples_per_plugin images=100 --logdir_spec tdip:results/phantom_3/low_res_as_cava_v1_10/TimedependentDIP,multires:results/phantom_3/low_res_as_cava_v1_10/MultiResFMLP

tensorboard --samples_per_plugin images=100 --logdir_spec multires:results/cava_v1/10/MultiResFMLP/validation/225/spatial_coordinate_scales


tensorboard --samples_per_plugin images=100 --logdir_spec multires:results/cava_v1/10/MultiResFMLP/validation/225/temporal_coordinate_scales

tensorboard --samples_per_plugin images=100 --logdir_spec multires:results/cava_v1/10/MultiResFMLP/validation/225/tv_regularization

tensorboard --samples_per_plugin images=100 --logdir_spec vae:results/cava_v1/11/default_vae,vaetv:results/cava_v1/11/default_vae_tv,tdip:results/cava_v1/11/TimedependentDIP

tensorboard --samples_per_plugin images=100 --logdir_spec vae:results/cava_v1/10/default_vae_tv
```