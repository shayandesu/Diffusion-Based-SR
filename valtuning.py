import torch.nn.functional as F
import torchmetrics
import torch
from torch import Tensor

class ValidityRatio(torchmetrics.Metric):
    full_state_update = False           # we update per-epoch, not per-batch
    higher_is_better = True             # larger ratio = better model

    def __init__(self, env, num_trials: int = 50):
        super().__init__()
        self.env = env
        self.num_trials = num_trials

        # two accumulators
        self.add_state("valid_gen", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_gen", default=torch.tensor(0.), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(self, model_outputs: Tensor):
        """
        model_outputs: (B, L) tensor of generated token IDs
        """
        total = model_outputs.size(0)
        valid = 0
        for out in model_outputs:
            idx_to_words = [self.env.equation_id2word[int(t)] for t in out]
            node = self.env.equation_encoder.decode(idx_to_words)
            if node is not None:
                valid += 1
        self.valid_gen += valid
        self.total_gen += total

    def compute(self):
        return self.valid_gen / (self.total_gen + 1e-8)


class ValidationLoss:
    def __init__(self, w2i):
        self.sign_indices = [w2i['+'], w2i['-']]
        self.mantissa_indices = [v for k,v in w2i.items() if k.startswith("N")]
        self.exp_indices = [v for k,v in w2i.items() if k.startswith("E")]
        self.math_constants = [w2i[v] for v in ['e', 'pi', 'euler_gamma']]
        self.vars = [v for k,v in w2i.items() if k.startswith("x_")]
        self.binaries = [w2i[v] for v in ['add', 'sub', 'mul', 'div', 'pow']]
        self.numerics = self.sign_indices + self.mantissa_indices + self.exp_indices
        self.constants = self.vars + self.math_constants + [36]
    
    
    def _diff_loss(self, logits):
        probs = F.softmax(logits, dim=-1)
        
        e_numerics = probs[:, :, self.numerics].sum(dim=-1).sum(dim=-1) / 3
        e_const = probs[:, :, self.constants].sum(dim=-1).sum(dim=-1)
        expected_binaries = probs[:, :, self.binaries].sum(dim=-1).sum(dim=-1)
        
        difference = (e_numerics + e_const - expected_binaries)
        
        return ((difference - 1.0) ** 2).mean()
    
    def loss(self, logits):
        return 0.01 * self._diff_loss(logits)