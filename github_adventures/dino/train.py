import argparse
import json
import pathlib

import timm
import torch
import torchvision.transforms as transforms
import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder

from evaluation import compute_embedding, compute_knn
from utils import DataAugmentation, Head, Loss, MultiCropWrapper, clip_gradients


def main():
    parser = argparse.ArgumentParser(
        "DINO training CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument(
        "-d", "--device", type=str, choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument("-l", "--logging-freq", type=int, default=200)
    parser.add_argument("--momentum-teacher", type=int, default=0.9995)
    parser.add_argument("-c", "--n-crops", type=int, default=4)
    parser.add_argument("-e", "--n-epochs", type=int, default=100)
    parser.add_argument("-o", "--out-dim", type=int, default=1024)
    parser.add_argument("-t", "--tensorboard-dir", type=str, default="logs")
    parser.add_argument("--clip-grad", type=float, default=2.0)
    parser.add_argument("--norm-last-layer", action="store_true")
    parser.add_argument("--batch-size-eval", type=int, default=64)
    parser.add_argument("--teacher-temp", type=float, default=0.04)
    parser.add_argument("--student-temp", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("-w", "--weight-decay", type=float, default=0.4)

    args = parser.parse_args()
    print(vars(args))
    # Parameters
    vit_name, dim = "vit_deit_small_patch16_224", 384
    path_dataset_train = pathlib.Path("data/imagenette2-320/train")
    path_dataset_val = pathlib.Path("data/imagenette2-320/val")
    path_labels = pathlib.Path("data/imagenette_labels.json")

    logging_path = pathlib.Path(args.tensorboard_dir)
    device = torch.device(args.device)

    n_workers = 4

    # Data related
    with path_labels.open("r") as f:
        label_mapping = json.load(f)

    transform_aug = DataAugmentation(size=224, n_local_crops=args.n_crops - 2)
    transform_plain = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Resize((224, 224)),
        ]
    )

    dataset_train_aug = ImageFolder(path_dataset_train, transform=transform_aug)
    dataset_train_plain = ImageFolder(path_dataset_train, transform=transform_plain)
    dataset_val_plain = ImageFolder(path_dataset_val, transform=transform_plain)

    if dataset_train_plain.classes != dataset_val_plain.classes:
        raise ValueError("Inconsistent classes")

    data_loader_train_aug = DataLoader(
        dataset_train_aug,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    data_loader_train_plain = DataLoader(
        dataset_train_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
    data_loader_val_plain = DataLoader(
        dataset_val_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        num_workers=n_workers,
    )
    data_loader_val_plain_subset = DataLoader(
        dataset_val_plain,
        batch_size=args.batch_size_eval,
        drop_last=False,
        sampler=SubsetRandomSampler(list(range(0, len(dataset_val_plain), 50))),
        num_workers=n_workers,
    )

    # Logging
    writer = SummaryWriter(logging_path)
    writer.add_text("arguments", json.dumps(vars(args)))

    # Neural network related
    student_vit = timm.create_model(vit_name, pretrained=args.pretrained)
    teacher_vit = timm.create_model(vit_name, pretrained=args.pretrained)

    student = MultiCropWrapper(
        student_vit,
        Head(
            dim,
            args.out_dim,
            norm_last_layer=args.norm_last_layer,
        ),
    )
    teacher = MultiCropWrapper(teacher_vit, Head(dim, args.out_dim))
    student, teacher = student.to(device), teacher.to(device)

    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # Loss related
    loss_inst = Loss(
        args.out_dim,
        teacher_temp=args.teacher_temp,
        student_temp=args.student_temp,
    ).to(device)
    lr = 0.0005 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    n_batches = len(dataset_train_aug) // args.batch_size
    best_acc = 0
    n_steps = 0

    for e in range(args.n_epochs):
        for i, (images, _) in tqdm.tqdm(
            enumerate(data_loader_train_aug), total=n_batches
        ):
            if n_steps % args.logging_freq == 0:
                student.eval()

                # Embedding
                embs, imgs, labels_ = compute_embedding(
                    student.backbone,
                    data_loader_val_plain_subset,
                )
                writer.add_embedding(
                    embs,
                    metadata=[label_mapping[l] for l in labels_],
                    label_img=imgs,
                    global_step=n_steps,
                    tag="embeddings",
                )

                # KNN
                current_acc = compute_knn(
                    student.backbone,
                    data_loader_train_plain,
                    data_loader_val_plain,
                )
                writer.add_scalar("knn-accuracy", current_acc, n_steps)
                if current_acc > best_acc:
                    torch.save(student, logging_path / "best_model.pth")
                    best_acc = current_acc

                student.train()

            images = [img.to(device) for img in images]

            teacher_output = teacher(images[:2])
            student_output = student(images)

            loss = loss_inst(student_output, teacher_output)

            optimizer.zero_grad()
            loss.backward()
            clip_gradients(student, args.clip_grad)
            optimizer.step()

            with torch.no_grad():
                for student_ps, teacher_ps in zip(
                    student.parameters(), teacher.parameters()
                ):
                    teacher_ps.data.mul_(args.momentum_teacher)
                    teacher_ps.data.add_(
                        (1 - args.momentum_teacher) * student_ps.detach().data
                    )

            writer.add_scalar("train_loss", loss, n_steps)

            n_steps += 1


if __name__ == "__main__":
    main()
