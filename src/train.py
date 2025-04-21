import os
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset.Pascal3DPlus import Normalize
from dataset.Pascal3DPlus import Pascal3DPlus
from dataset.Pascal3DPlus import ToTensor
from lib.get_n_list import get_n_list
from models.FeatureBank import mask_remove_near
from models.FeatureBank import FeatureBank
from models.KeypointRepresentationNet import NetE2E
from tqdm import trange
from lib.config import load_config, parse_args
import torch.nn as nn
from torch.nn import functional as F
import csv
import torch.distributed as dist
from siglip.loss import SigLipLoss

args = parse_args()
config = load_config(args, load_default_config=False, log_info=False)
n_gpus = torch.cuda.device_count()
local_size = [config.model.local_size, config.model.local_size]

bank_set = []
dataloader_set = []
n_list_set = []
mesh_path_set = []


if config.dataset.paths.mesh:
    for class_ in config.dataset.classes:
        mesh_path = Path(config.dataset.paths.root, config.dataset.paths.mesh, class_)
        mesh_path_set.append(mesh_path)
        n_list = get_n_list(mesh_path)
        n_list_set.append(n_list[0])


os.makedirs(config.save_dir, exist_ok=True)

net = NetE2E(
    net_type=config.model.backbone,
    local_size=local_size,
    output_dimension=config.training.d_feature,
    n_noise_points=config.model.num_noise,
    pretrain=True,
    noise_on_mask=False,
)
net.train()
if config.training.separate_bank:
    net = torch.nn.paralled.DataParallel(
        net.cuda(), device_ids=[i for i in range(n_gpus - 1)]
    )
else:
    net = torch.nn.parallel.DataParallel(net.cuda())


transforms = transforms.Compose(
    [
        ToTensor(),
        Normalize(),
    ],
)

mesh_path = mesh_path_set[0]
max_n = max(n_list_set)
fbank = FeatureBank(
    inputSize=config.training.d_feature,
    outputSize=len(config.dataset.classes) * max_n
    + config.model.num_noise * config.model.max_group,
    num_noise=config.model.num_noise,
    num_pos=len(config.dataset.classes) * max_n,
    momentum=config.model.adj_momentum,
    nb_classes=len(config.dataset.classes),
)
fbank = fbank.cuda()

dataset = Pascal3DPlus(
    transforms=transforms, max_n=max_n, occlusion="", config=config.dataset
)

shared_dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.workers,
)

t_prime = nn.Parameter(torch.log(torch.tensor(10.0)))
b = nn.Parameter(torch.tensor(-10.0))
logit_scale = torch.exp(t_prime)
logit_bias = b

criterion = SigLipLoss()

iter_num = 0
optim = torch.optim.Adam(
    list(net.parameters() + [t_prime, b]),
    lr=config.training.lr,
    weight_decay=config.training.weight_decay,
)
last_device = "cuda:%d" % (n_gpus - 1)
fbank = fbank.cuda(last_device)

pad_index = []
for i in range(len(config.dataset.classes)):
    num = (max_n * (i + 1)) - (max_n * i + n_list_set[i])
    for j in range(num):
        n = max_n * i + n_list_set[i] + j
        pad_index.append(n)

pad_index = torch.tensor(pad_index, dtype=torch.long)
zeros = torch.zeros(
    config.training.batch_size,
    max_n,
    max_n * len(config.dataset.classes),
    dtype=torch.float32,
).to(last_device)

experiment_name = "siglip_loss"
csv_file = f"{config.save_dir}/training_log_{experiment_name}.csv"


def log_training_metrics(
    iter_num,
    epoch,
    loss_main,
    loss_reg,
    csv_file="training_log.csv",
    print_to_console=True,
):
    """
    Logs training metrics to a CSV file with optional console printing.
    """
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(csv_file)

    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare the row
    row = [timestamp, iter_num, epoch, f"{loss_main:.5f}", f"{loss_reg:.5f}"]

    # Write to CSV
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header if it's a new file
        if not file_exists:
            writer.writerow(["timestamp", "n_iter", "epoch", "loss", "loss_reg"])

        # Write data row
        writer.writerow(row)

    # Optionally, print to console
    if print_to_console:
        print(
            "timestamp",
            timestamp,
            "n_iter",
            iter_num,
            "epoch",
            epoch,
            "loss",
            f"{loss_main:.5f}",
            "loss_reg",
            f"{loss_reg:.5f}",
        )


def save_checkpoint(state, filename):
    file = os.path.join(config.save_dir, filename)
    torch.save(state, file)


print("Start Training!")
for epoch in trange(config.training.total_epochs):
    if (epoch - 1) % config.training.update_lr_epoch_n == 0:
        lr = config.training.lr * config.training.update_lr_
        for param_group in optim.param_groups:
            param_group["lr"] = lr

    y_num = max_n
    for i, sample in enumerate(shared_dataloader):
        img, keypoint, iskpvisible, box_obj, img_label = (
            sample["img"],
            sample["kp"],
            sample["iskpvisible"],
            sample["box_obj"],
            sample["label"],
        )
        # obj_mask = sample["obj_mask"]
        index = sample["y_idx"]
        index = index.cuda()

        img = img.cuda()
        keypoint = keypoint.cuda()
        iskpvisible = iskpvisible.cuda()

        iskpvisible_float = iskpvisible
        iskpvisible = iskpvisible.type(torch.bool).to(iskpvisible.device)

        # obj_mask = obj_mask.cuda()
        img_label = img_label.cuda()

        # feature is of shape [batch, -1, d_feature (128 as setted)]
        image_features = net.forward(
            img, keypoint_positions=keypoint
        )  # , obj_mask=1 - obj_mask)

        bank_features = fbank.features

        # flatten and mask out invisible vertices
        B, K, D = image_features.shape
        flat_img_feats = image_features.view(B * K, D)
        flat_mask = iskpvisible.view(-1)
        image_feats = flat_img_feats[flat_mask]

        # get matching features from FeatureBank
        flat_ids = index.view(-1)
        target_ids = flat_ids[flat_mask]
        bank_feats = bank_features[target_ids]

        loss = criterion(bank_feats, image_feats, target_ids, logit_scale, logit_bias)

        loss_main = loss.item()
        fbank.forward_siglip(image_features, index, iskpvisible_float, img_label)

        if config.model.num_noise > 0:
            loss_reg = torch.mean(noise_sim) * 0.1
            # The loss of noise
            loss += loss_reg
        else:
            loss_reg = torch.zeros(1)

        loss.backward()
        if iter_num % config.training.accumulate == 0:
            optim.step()
            optim.zero_grad()
            log_training_metrics(
                iter_num, epoch, loss_main, loss_reg.item(), csv_file=csv_file
            )

        iter_num += 1

    if (epoch + 1) % 5 == 0:
        save_checkpoint(
            {
                "state": net.state_dict(),
                "memory": fbank.memory,
                "timestamp": int(datetime.timestamp(datetime.now())),
                "args": args,
            },
            f"{experiment_name}_classification_saved_model_{epoch + 1}.pth",
        )
