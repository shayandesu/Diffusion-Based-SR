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
import rich.syntax
import rich.tree
import fsspec

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)
# import copy

# class ValidityRatioCallback(L.Callback):
#     def __init__(self, env, num_trials=2, num_steps=200):
#         super().__init__()
#         self.env = env
#         self.num_trials = num_trials
#         self.num_steps = num_steps

#     @torch.no_grad()
#     def on_train_epoch_end(self, trainer, pl_module):
#         if hasattr(trainer, "global_rank") and trainer.global_rank != 0:
#             return  # only run on rank 0 in DDP

#         model_copy = copy.deepcopy(pl_module).to(pl_module.device)
#         if hasattr(model_copy, "ema"):
#             model_copy.ema.copy_to(model_copy.parameters())

#         model_copy.eval()

#         total_gen = valid_gen = 0
#         for _ in range(self.num_trials):
#             samples = model_copy._sample(self.num_steps).detach().cpu()  # (B, L)
#             total_gen += samples.shape[0]

#             for seq in samples:
#                 seq = seq.tolist()
#                 words = [self.env.equation_id2word[t] for t in seq]
#                 if self.env.equation_encoder.decode(words) is not None:
#                     valid_gen += 1

#         ratio = valid_gen / max(total_gen, 1)
#         pl_module.log("val/valid_ratio", ratio, prog_bar=True, sync_dist=True)

#         del model_copy
#         torch.cuda.empty_cache()



@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig,
    resolve: bool = True,
    save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
              config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


@hydra.main(version_base=None, config_path='hyperparameters/configs',
            config_name='config')
def main(config):
    # Load SNIP params
    with open("/home/xulei/shayan/SR/DDSR/hyperparameters/params.pkl", 'rb') as p:
        params = pickle.load(p)
    
    L.seed_everything(config.seed)
    params.size = config.size
    params.batch_size = config.batch_size
    params.conditioning = config.model.text_conditioning
    params.latent_dim = 512
    # _print_config(config)
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
    
    tk = Tokenizer(params, 200)
    config.model.smiles_length=tk.max_len
    env = FunctionEnvironment(params, tk)
    model = Diffusion(config, tk, env)
    train_dataloader = env.create_train_iterator(params.tasks, None, params)
    params.size=1024
    valid_dataloader = env.create_train_iterator(params.tasks, None, params)
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
            
    # callbacks.append(ValidityRatioCallback(env))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_dataloader, valid_dataloader, ckpt_path=None)
    



if __name__ == "__main__":
    main()