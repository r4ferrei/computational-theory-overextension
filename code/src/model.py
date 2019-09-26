import os

import numpy as np
import torch

# NOTE: following inputs and outputs are PyTorch, not NumPy.

def compute_likelihood(prod_sense_dist, kernel_widths):
    p, s, d = prod_sense_dist.shape
    assert(
            ((p,) == kernel_widths.shape) or
            ((1,) == kernel_widths.shape))

    # NOTE: 1 for Manhattan+exponential, 2 for Euclidean+Gaussian
    KERNEL_EXP = 2

    curr = prod_sense_dist
    curr = torch.norm(curr, p=KERNEL_EXP, dim=2)
    curr = torch.div(curr, torch.reshape(kernel_widths, (-1,1)))
    curr = torch.exp(-torch.pow(curr, KERNEL_EXP))
    return curr

def block_prods(prod_sense_dist, prod_indices):
    zero = torch.zeros_like(prod_sense_dist)
    keep = torch.ones_like(prod_sense_dist, dtype=torch.uint8)
    for col, row in enumerate(prod_indices):
        if row is not None and row != -1:
            keep[row, col] = 0
    return torch.where(keep, prod_sense_dist, zero)

def apply_prior(lik, priors):
    p, s = lik.shape
    assert(
            ((p,) == priors.shape) or
            ((1,) == priors.shape))

    curr = lik
    curr = torch.mul(curr, priors.reshape(-1,1))
    return curr

def rel_freqs_to_priors(rel_freqs, uniform=False):
    if uniform:
        return torch.ones((1,), dtype=rel_freqs.dtype, device=rel_freqs.device)
    else:
        return rel_freqs

def compute_kernel_widths(h):
    return h

def _normalized_posterior(
        prod_sense_dist, identity_prod_indices,
        kernel_widths, priors, full_dist_matrix,
        allow_identity_prod = False,
        baseline            = False):
    if baseline:
        lik = torch.ones_like(prod_sense_dist.sum(dim=2))
    else:
        lik = compute_likelihood(prod_sense_dist, kernel_widths)

    post = apply_prior(lik, priors)
    if not allow_identity_prod:
        post = block_prods(post, identity_prod_indices)
    post = torch.div(post, torch.sum(post, dim=0))
    return post

# NOTE: identity_prod_indices can contain None/-1 items when sense is not even
# in full vocabulary. This is ok, because we only need this to hide these
# items from the optimization.
def production_nll(
        prod_sense_dist, child_prod_indices, identity_prod_indices,
        kernel_widths, priors, full_dist_matrix):
    post = _normalized_posterior(prod_sense_dist, identity_prod_indices,
            kernel_widths, priors, full_dist_matrix)

    nll = 0
    for col, row in enumerate(child_prod_indices):
        nll += -torch.log(post[row, col])
    return nll

def predict_production_ranks_and_posteriors(
        prod_sense_dist, identity_prod_indices,
        kernel_widths, priors, full_dist_matrix,
        child_prod_indices,
        allow_identity_prod  = False,
        target_identity_prod = False,
        get_full_matrix      = False,
        baseline             = False):

    post = _normalized_posterior(prod_sense_dist, identity_prod_indices,
            kernel_widths, priors, full_dist_matrix,
            allow_identity_prod = allow_identity_prod,
            baseline            = baseline)

    ranks = torch.zeros((prod_sense_dist.shape[1],), dtype=torch.int32)
    posts = torch.zeros_like(ranks, dtype=prod_sense_dist.dtype)
    for i in range(len(ranks)):
        if target_identity_prod:
            if (identity_prod_indices[i] is None or
                    identity_prod_indices[i] == -1):
                ranks[i] = 1000
                posts[i] = 0
            else:
                ranks[i] = torch.sum(
                        post[:,i] >= post[identity_prod_indices[i],i])
                posts[i] = post[identity_prod_indices[i], i]
        else: # target child prod
            ranks[i] = torch.sum(post[:,i] >= post[child_prod_indices[i],i])
            posts[i] = post[child_prod_indices[i], i]

    if get_full_matrix:
        return ranks, posts, post
    else:
        return ranks, posts

def predict_top_k_prods(
        prod_sense_dist, identity_prod_indices,
        kernel_widths, priors, full_dist_matrix,
        allow_identity_prod=False, k=5):

    post = _normalized_posterior(prod_sense_dist, identity_prod_indices,
            kernel_widths, priors, full_dist_matrix,
            allow_identity_prod=allow_identity_prod)

    prods = [[] for _ in range(prod_sense_dist.shape[1])]
    posts = [[] for _ in range(prod_sense_dist.shape[1])]
    for i in range(len(prods)):
        _values, indices = torch.sort(post[:,i], descending=True)
        for ind in indices[:k]:
            posts[i].append(post[ind, i])
            prods[i].append(int(ind.detach().cpu()))

    return prods, posts
