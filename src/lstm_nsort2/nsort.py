import torch


def bl_matmul(mat_a, mat_b):
    return torch.einsum('mij,jk->mik', mat_a, mat_b)


def compute_permu_matrix(s: torch.FloatTensor, tau=1):
    """
    Calculates neuralsort relaxed permutation matrix.

    Args:
        s:              Vector of scores
        tau:            Temperature of relaxation

    Returns:
        FloatTensor:    Returns permutation matrix.

    """

    mat_as = s - s.permute(0, 2, 1)
    mat_as = torch.abs(mat_as)
    n = s.shape[1]
    one = torch.ones(n, 1).to(device)
    b = _bl_matmul(mat_as, one @ one.transpose(0, 1))
    k = torch.arange(n) + 1
    d = (n + 1 - 2 * k).float().detach().requires_grad_(True).unsqueeze(0).to(device)
    c = _bl_matmul(s, d)
    mat_p = (c - b).permute(0, 2, 1)
    mat_p = F.softmax(mat_p / tau, -1)

    return mat_p


def prop_any_correct(p1, p2):
    """
    Calculates individual number of items sorted correctly.

    Args:
        p1:             Input tensor p hat
        p2:             Label tensor p true

    Returns:
        FloatTensor:    Returns number of correctly sorted items.

    """

    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2).float()
    correct = torch.mean(eq, axis=-1)
    return torch.mean(correct)


def prop_correct(p1, p2):
    """
    Calculates number of permutations with all items correctly sorted.

    Args:
        p1:             Input tensor p hat
        p2:             Label tensor p true

    Returns:
        FloatTensor:    Returns number of p hats that are totally correct.

    """

    z1 = torch.argmax(p1, axis=-1)
    z2 = torch.argmax(p2, axis=-1)
    eq = torch.eq(z1, z2)
    correct = torch.all(eq, axis=-1).float()
    return torch.sum(correct)
