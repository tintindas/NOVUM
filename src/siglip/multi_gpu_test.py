import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from .loss import SigLipLoss  # Adjust this import path to your actual module


def setup():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank, dist.get_world_size()


def cleanup():
    dist.destroy_process_group()


def main():
    rank, world_size = setup()

    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42 + rank)

    # Dummy features
    feat_dim = 64
    num_feats = 8

    image_feats = torch.randn(num_feats, feat_dim, device=device, requires_grad=True)
    bank_feats = torch.randn(num_feats, feat_dim, device=device)
    ids = torch.arange(num_feats, device=device)

    # logit scale and bias
    t_prime = torch.nn.Parameter(torch.log(torch.tensor(10.0, device=device)))
    b = torch.nn.Parameter(torch.tensor(-10.0, device=device))
    logit_scale = torch.exp(t_prime)
    logit_bias = b

    # Loss instantiation and execution
    criterion = SigLipLoss(rank=rank, world_size=world_size)
    loss = criterion(image_feats, bank_feats, ids, logit_scale, logit_bias)

    print(f"[Rank {rank}] Loss: {loss.item()}")

    loss.backward()
    print(f"[Rank {rank}] Backward successful")

    cleanup()


if __name__ == "__main__":
    main()
