import torch
import torch.nn.functional as F

p_hat = torch.tensor([[[0.2852, 0.0281, 0.6867],
                       [0.5327, 0.2461, 0.2212],
                       [0.3090, 0.6688, 0.0221]]])

p_true = torch.tensor([[[0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0]]])

losses = -torch.sum(p_true * torch.log(p_hat + 1e-20), dim=-1).mean()

# loss =

print(losses)
