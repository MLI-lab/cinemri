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
tensorboard --samples_per_plugin images=100 --logdir_spec kfmlp:results/cava_v1_static/10/subsampled_single_frame/KFMLP/

tensorboard --samples_per_plugin images=100 --logdir_spec kfmlp:results/cava_v1_static/10/subsampled_single_frame/FMLP/

tensorboard --samples_per_plugin images=100 --logdir_spec kfmlp:results/cava_v1/10/KFMLP/validation/225/

```