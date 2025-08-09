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

omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def evaluate(model: Diffusion, env: FunctionEnvironment, ds: EnvDataset, num_trials: int = 10):
    for i in range(num_trials):
        org = ds.generate_sample()
        x_to_fit = org['x_to_fit']
        y_to_fit = org['y_to_fit']
        embeds = ds._preprocess_sample(org)['text_embeddings'].unsqueeze(0).to('cuda')
        
        model_error = np.inf
        model_outputs = model.restore_model_and_sample(200, text_embeddings=embeds)
        for out in model_outputs:
            if out is None:
                continue
            
            idx_to_words = [env.equation_id2word[int(term)] for term in out]
            node = env.equation_encoder.decode(idx_to_words)
            try:
                vals = node.val(x_to_fit)
            except:
                vals = None
            
            if vals is None:
                continue
            
            error = np.log(np.mean((vals - y_to_fit) ** 2))
            model_error = min(model_error, error)
        
        random_error = None
        while random_error is None:
            sample = ds.generate_sample()
            try:
                vals = sample['tree'].val(x_to_fit)
            except:
                vals = None
            
            if vals is None:
                continue
            
            random_error = np.log(np.mean((vals - y_to_fit) ** 2))
    
        print(f"Processed sample {i+1}/{num_trials}. Model Error: {model_error:.4f}, Random Error: {random_error:.4f}")
            

@hydra.main(version_base=None, config_path='hyperparameters/configs',
            config_name='config')
def main(config):
    config.model.smiles_length=200
    # Load SNIP params
    with open("/home/xulei/shayan/SR/DDSR/hyperparameters/params.pkl", 'rb') as p:
        params = pickle.load(p)
        
    params.size=1000
        
    L.seed_everything(config.seed)
    tk = Tokenizer(params)
    model = Diffusion.load_from_checkpoint(
        "/home/xulei/shayan/SR/DDSR/outputs/2025.08.06/113517/checkpoints/best.ckpt",
        tokenizer=tk,
        config=config
        )
    
    env = FunctionEnvironment(params, tk)
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
    
    evaluate(model, env, dataset)    
    


if __name__ == "__main__":
    main()