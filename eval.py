import hydra
import pickle
import lightning as L
from tokenizer import Tokenizer
from diffusion import Diffusion
import omegaconf
from snip.envs.environment import FunctionEnvironment, EnvDataset
import torch
import os
import numpy as np
from tqdm import tqdm

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def validity_evaluator(model: Diffusion, env: FunctionEnvironment, ds: EnvDataset = None, num_trials: int = 100):
    total_gen = 0
    valid_gen = 0
    ratio = 0.0
    print("Starting Generation")
    pbar = tqdm(range(num_trials))
    for i in pbar:
        model_outputs = model.restore_model_and_sample(200)
        total_gen += model_outputs.shape[0]
        if i == 0:
            print(f"Output Shape: {model_outputs.shape}")
            print(f"Sample Output: {model_outputs[0]}")
        for out in model_outputs:
            if out is None:
                print("Wow")
                continue
            
            idx_to_words = [env.equation_id2word[int(term)] for term in out]
            node = env.equation_encoder.decode(idx_to_words)
            if node is not None:
                valid_gen +=1
        
        ratio = valid_gen/total_gen
        pbar.set_description(f"{valid_gen}/{total_gen} ({ratio:.3f})")
    
    print(f"Finished Trials. Valids: {valid_gen}, Total: {total_gen}, Validity Ratio: {ratio:.3f}")
                
            

@hydra.main(version_base=None, config_path='hyperparameters/configs',
            config_name='config')
def main(config):
    config.model.smiles_length=200
    # Load SNIP params
    with open("/home/xulei/shayan/SR/DDSR/hyperparameters/params.pkl", 'rb') as p:
        params = pickle.load(p)
        
    params.size=1000
    params.conditioning = config.model.text_conditioning
        
    L.seed_everything(config.seed)
    tk = Tokenizer(params)
    print(f"<EOS>: {tk.eos_token_id} | <PAD>: {tk.pad_token_id}")

    if config.checkpointing.resume_from_ckpt:
        ckpt = config.checkpointing.resume_ckpt_path
        model = Diffusion.load_from_checkpoint(
            ckpt,
            tokenizer=tk,
            config=config
        )
        print(f"Successfully loaded model from {ckpt}")
    else:
        model = Diffusion(config, tk).to('cpu')
    
    env = FunctionEnvironment(params, tk)
    dataset = None
    if config.model.text_conditioning:
        print("Generating Dataset.")
        dataset = EnvDataset(
                    env,
                    params.tasks,
                    train=True,
                    tokenizer=env.tokenizer,
                    skip=params.queue_strategy is not None,
                    params=params,
                    path=None,
                    size=1000
                )

        dataset.init_rng()
    
    validity_evaluator(model, env, dataset)    
    


if __name__ == "__main__":
    main()