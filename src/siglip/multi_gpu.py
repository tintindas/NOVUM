import os
import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    global_rank = dist.get_rank()

    print(f"Global Rank {global_rank} using GPU {torch.cuda.current_device()}")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    main()
