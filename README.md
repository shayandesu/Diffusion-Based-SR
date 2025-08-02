# Diffusion-Based-SR

### Creating the Environment
Activate the conda environment using the commands below
```
conda env create -f environment.yaml
conda activate ddsr
```

### Download SNIP Pre-trained Models
Run the following command and the pipeline will automatically used the pretrained data point encoder and embedder.
```
gdown --folder https://drive.google.com/drive/folders/1oGVQPAuTwWQnhX_pxN3OdKDt9-rmCfV3
```

### Run the Training
```
python main.py
```

### Model Modification
Details on the architecture and hyperparameters of the model are available at `hyperparameteres/configs` and can be modified.
Batch size and dataset size can be set using `params.batch_size = ...` and `params.size = ...` respectively.

### Checkpointing
After each epoch, the best and also the last weights are saved at `outputs/openwebtext/.../best.ckpt`/`outputs/openwebtext/.../last.ckpt`.
The metric used to evaluate and find the best model weights can be set at `hyperparameters/configs/callbacks/checkpoint_monitor.yaml` by changing `monitor` (currently set to NLL loss on training).
