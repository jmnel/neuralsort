import torch
import torch.nn as nn
import torch.nn.functional as F


def _bl_matmul(A, B):
    return torch.einsum('mij,jk->mik', A, B)


torch.manual_seed(0)
batch_size = 2
seq_len = 5
stitch_len = 4


def compute_permu_matrix(s: torch.FloatTensor, tau=1):
    A_s = s - s.permute(0, 2, 1)
    A_s = abs(A_s)
    print(A_s.shape)
    n = s.shape[1]
    one = torch.ones(n, 1)
    B = _bl_matmul(A_s, one @ one.transpose(0, 1))
    K = torch.arange(n) + 1
    C = _bl_matmul(s,
                   torch.tensor(n + 1 - 2 * K
                                ).unsqueeze(0).float())
    P = (C - B).permute(0, 2, 1)
    P = F.softmax(P / tau, -1)

    return P


s = torch.randn(batch_size, seq_len, 1)

foo = compute_permu_matrix(s)
