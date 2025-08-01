import pickle
import os
import torch
import hydra
from tokenizer import Tokenizer
from diffusion import Diffusion
import omegaconf
from snip.envs.environment import FunctionEnvironment
import utils
import lightning as L

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


@hydra.main(version_base=None, config_path='hyperparameters/configs',
            config_name='config')
def main(config):
    # Load SNIP params
    with open("/home/xulei/shayan/SR/DDSR/hyperparameters/params.pkl", 'rb') as p:
        params = pickle.load(p)
    
    L.seed_everything(config.seed)
    params.size = 1000000
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
        config=omegaconf.OmegaConf.to_object(config),
        ** config.wandb)
        
    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
          config.checkpointing.resume_ckpt_path)):
      ckpt_path = config.checkpointing.resume_ckpt_path
    else:
      ckpt_path = None
    
    tk = Tokenizer(params)
    model = Diffusion(config, tk)
    env = FunctionEnvironment(params, tk)
    train_dataloader = env.create_train_iterator(params.tasks, None, params)
    valid_dataloader = None
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
            
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=None)
    



if __name__ == "__main__":
    main()