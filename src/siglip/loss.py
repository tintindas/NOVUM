import torch
import torch.nn as nn
from torch.nn import functional as F


# Handling backprop across multiple GPUs
def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (
            NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),
        )


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class SigLipLoss(nn.Module):
    """Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """

    def __init__(
        self,
        rank=0,
        world_size=1,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

    def _loss(self, bank_feats, image_feats, target_ids, logit_scale, logit_bias):
        # get logits
        # bank_feats: [N, D]
        # image_feats: [N, D]
        logits = logit_scale * (bank_feats @ image_feats.T)  # [N,N]
        if logit_bias is not None:
            logits += logit_bias

        # get labels
        eq_mask = target_ids.unsqueeze(1) == target_ids.unsqueeze(0)  # [N, N]
        labels = torch.where(eq_mask, 1.0, -1.0).to(logits.device, logits.dtype)

        loss = -F.logsigmoid(labels * logits).mean()

        return loss

    def forward(
        self, bank_feats, image_feats, target_ids, logit_scale, logit_bias=None
    ):
        z_bank_feats = F.normalize(bank_feats, p=2, dim=1)
        z_image_feats = F.normalize(image_feats, p=2, dim=1)
        loss = self._loss(
            z_bank_feats, z_image_feats, target_ids, logit_scale, logit_bias
        )

        if self.world_size > 1:
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size

            image_feats_to_right = image_feats
            for i in range(self.world_size - 1):
                image_feats_from_left = neighbour_exchange_with_grad(
                    left_rank, right_rank, image_feats_to_right
                )

                loss += self._loss(
                    bank_feats,
                    image_feats_from_left,
                    logit_scale,
                    logit_bias,
                )
                image_feats_to_right = image_feats_from_left

        return loss
