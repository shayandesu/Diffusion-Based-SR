# Diffusion-Based-SR

## Creating the Environment
Activate the conda environment using the commands below
```
conda env create -f environment.yaml
conda activate ddsr
```

## Download SNIP Pre-trained Models
Run the following command and the pipeline will automatically used the pretrained data point encoder and embedder.
```
gdown --folder https://drive.google.com/drive/folders/1oGVQPAuTwWQnhX_pxN3OdKDt9-rmCfV3
```

## Run the Training
### Setup 1.
Default (subs)
```
HYDRA_FULL_ERROR=1 python main.py model=large seed=15 batch_size=64
```
```
HYDRA_FULL_ERROR=1 python main.py model=large seed=15 batch_size=128
```
Medium-sized model:
```
HYDRA_FULL_ERROR=1 python main.py model=medium seed=15
```


### Setup 2.
subs - discrete time
```
HYDRA_FULL_ERROR=1 python main.py model=large T=1000 seed=15 batch_size=64
```
```
HYDRA_FULL_ERROR=1 python main.py model=large T=1000 seed=15 batch_size=128
```
Medium-sized model:
```
HYDRA_FULL_ERROR=1 python main.py model=medium T=1000 seed=15
```

### Setup 3.
d3pm
```
HYDRA_FULL_ERROR=1 python main.py model=large T=1000 parameterization=d3pm time_conditioning=True subs_masking=True seed=15 batch_size=64
```

```
HYDRA_FULL_ERROR=1 python main.py model=large T=1000 parameterization=d3pm time_conditioning=True subs_masking=True seed=15 batch_size=128
```
Medium-sized model:
```
HYDRA_FULL_ERROR=1 python main.py model=medium T=1000 parameterization=d3pm time_conditioning=True subs_masking=True seed=15
```

## Model Modification
Details on the architecture and hyperparameters of the model are available at `hyperparameteres/configs` and can be modified.

## Checkpointing
After each epoch, the best and also the last weights are saved at `outputs/.../best.ckpt`/`outputs/.../last.ckpt`.
The metric used to evaluate and find the best model weights can be set at `hyperparameters/configs/callbacks/checkpoint_monitor.yaml` by changing `monitor` (currently set to NLL loss on training).
