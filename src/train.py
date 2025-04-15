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
from torch.nn import functional as F
import csv

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
    net = torch.nn.DataParallel(net.cuda(), device_ids=[i for i in range(n_gpus - 1)])
else:
    net = torch.nn.DataParallel(net.cuda())


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

criterion = torch.nn.CrossEntropyLoss(reduction="none").cuda()

iter_num = 0
optim = torch.optim.Adam(
    net.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
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

experiment_name = "manual_sig_loss"
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

    Parameters:
    - iter_num (int): Current iteration number.
    - epoch (int): Current epoch number.
    - loss_main (float): Main loss value.
    - loss_reg (float): Regularization loss value.
    - csv_file (str): Path to the CSV file.
    - print_to_console (bool): Whether to also print the log to console.
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

        img = img.cuda()
        keypoint = keypoint.cuda()
        iskpvisible = iskpvisible.cuda()
        # obj_mask = obj_mask.cuda()
        img_label = img_label.cuda()

        # feature is of shape [batch, -1, d_feature (128 as setted)]
        features = net.forward(
            img, keypoint_positions=keypoint
        )  # , obj_mask=1 - obj_mask)

        # similarity: [n, k, l]
        if config.training.separate_bank:
            similarity, y_idx, noise_sim, label_onehot = fbank(
                features.to(last_device),
                index.to(last_device),
                iskpvisible.to(last_device),
                img_label.to(last_device),
            )
        else:
            similarity, y_idx, noise_sim, label_onehot = fbank(
                features,
                index.cuda(),
                iskpvisible,
                img_label,
            )

        similarity /= config.training.T

        # make near vertice large value for CE, remove effect of near vertices.
        mask_distance_legal = mask_remove_near(
            keypoint,
            thr=config.training.distance_thr,
            num_neg=config.model.num_noise * config.model.max_group,
            img_label=img_label,
            pad_index=pad_index,
            nb_classes=len(config.dataset.classes),
            zeros=zeros,
            dtype_template=similarity,
            neg_weight=config.training.weight_noise,
        )

        iskpvisible_float = iskpvisible
        iskpvisible = iskpvisible.type(torch.bool).to(iskpvisible.device)

        logits = similarity.view(-1, similarity.shape[2]) - mask_distance_legal.view(
            -1, similarity.shape[2]
        )
        iskpvisible_flat = iskpvisible.view(-1)
        logits = logits[iskpvisible_flat, :]

        target = y_idx.view(-1)[iskpvisible_flat]

        log_probs = F.log_softmax(logits, dim=1)  # (N_visible, V)

        labels_onehot = torch.zeros_like(log_probs)
        labels_onehot.scatter_(1, target.unsqueeze(1), 1.0)

        per_example_loss = -torch.sum(log_probs * labels_onehot, dim=1)  # (N_visible,)

        loss = per_example_loss.mean()
        loss_main = loss.item()

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
