

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
import wandb
from test import test
from net import Net
from optimizer import Optimizer


def main(
    args,
    bs=512,
    img_size=32,
    resume=False,
    n_epochs=200,
    patch=4,
    dimhead=512,
    convkernel=8,
    num_classes=10,
    num_workers=4,
    dataset="cifar10",
    weights_from="",
    weights_to="",
    net="res18",
    use_amp=False,
    aug=True,
    opt="adam",
    artifact=None,
    schlr='cosine',
):

    usewandb = args.wandb
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_acc = 0.0  # best test accuracy
    start_epoch = 0

    # Prepare data
    print("==> Preparing data..")

    # Set image size based on the model used
    if net in [
        "cait",
        "simplevit",
        "vit_small",
        "vit_tiny",
        "cait_small",
        "swin_small",
        "swin_base",
        "swin_large",
        "swin_tiny",
        "gcvit_tiny",
        "gcit_small",
        "gcit_base",
        "gcit_large",
    ]:
        img_size = 224

    elif net in ["swinmlp_tiny_c6", "swinmlp_tiny_c12", "swinmlp_tiny_c24"]:
        img_size = 256

    # # Set number of classes and data transforms based on the dataset used
    if dataset == "cifar10":
        num_classes = 10

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        # Add RandAugment with N, M(hyperparameter)
        if aug:
            N = 2
            M = 14
            transform_train.transforms.insert(0, RandAugment(N, M))

        # Prepare dataset
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    elif dataset == "cifar100":
        num_classes = 100
        # train-test sampler need to be implemented

    elif dataset == "imagenet":
        num_classes = 1000
        # train-test sampler need to be implemented

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True, num_workers=num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=bs, shuffle=False, num_workers=num_workers
    )

    # Model factory
    print("==> Building model..")

    net = Net(
        args,
        img_size=img_size,
        bs=bs,
        num_classes=num_classes,
        patch=patch,
        dimhead=dimhead,
        convkernel=convkernel,
    )

    # For Multi-GPU
    if "cuda" in device:
        print(device)
        print("using data parallel")
        net = torch.nn.DataParallel(net)  # make parallel
        cudnn.benchmark = True

    optimizer = Optimizer(net, opt, args)

    if resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        if weights_from != "":
            checkpoint = torch.load(weights_from)
        else:
            try:
                artifact = wandb.use_artifact(
                    "iut-hert/PyTorch-Cifar-10/run_" + args.net + ":latest",
                    type="model",
                )
                artifact_dir = artifact.download()
                checkpoint = torch.load(
                    artifact_dir + "/wandb_best_{}.pt".format(args.net)
                )
            except:
                print("No checkpoint found")
                wandb.finish()
                exit()
        net.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint["epoch"] + 1
        if checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer loaded")

        del checkpoint  # current, saved

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    # use cosine scheduling
    if schlr == "cosine":
         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    if schlr == "reduceonplateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

    # Training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    list_loss = []
    list_acc = []

    if usewandb:
        wandb.watch(net)

    start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = "output/" + args.net + "/" + start_time + "/"

    if not os.path.exists(path):
        os.makedirs(path)

    net.cuda()
    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(trainloader),
                "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                % (
                    train_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    correct,
                    total,
                ),
            )

        trainloss = train_loss / (batch_idx + 1)

        # Evaluation
        val_loss, acc, best_acc = test(
            epoch, net, testloader, device, criterion, optimizer, scaler, best_acc, args
        )

        list_loss.append(val_loss)
        list_acc.append(acc)

        # Log training..
        if usewandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": trainloss,
                    "val_loss": val_loss,
                    "val_acc": acc,
                    "train_acc": 100.0 * correct / total,
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_time": time.time() - start,
                }
            )

        # Write out csv..
        with open(f"log/log_{args.net}_patch{args.patch}.csv", "w") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)

        checkpoint = {
            "epoch": epoch,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(checkpoint, path + "latest.pt")

        if acc == best_acc:
            checkpoint["optimizer"] = None
            torch.save(checkpoint, path + "best.pt")

        if schlr == "cosine":
            scheduler.step()
        elif schlr == "reduceonplateau":
            scheduler.step(val_loss)

    # writeout wandb
    if usewandb:
        model_artifact = wandb.Artifact(
            "run_" + args.net,
            type="model",
            metadata={
                "original_url": str(path),
                "epochs_trained": epoch + 1,
                "total_epochs": args.n_epochs,
                "best_acc": best_acc,
            },
        )
        # model_artifact.add_file(path + 'latest.pt', name="wandb_latest_{}_lr{}.pt".format(args.net, args.lr))
        model_artifact.add_file(
            path + "best.pt", name="wandb_best_{}.pt".format(args.net)
        )
        wandb.log_artifact(model_artifact)
        wandb.finish()


if __name__ == "__main__":
    # parsers
    parser = argparse.ArgumentParser(description="PyTorch CV Models Training")
    parser.add_argument(
        "--lr", default=1e-4, type=float, help="learning rate"
    )  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument(
        "--resume", "-r", action="store_true", help="resume from checkpoint"
    )
    parser.add_argument("--opt", default="adam", type=str, help="optimizer")
    parser.add_argument("--noaug", action="store_true", help="disable use randomaug")
    parser.add_argument(
        "--noamp",
        action="store_true",
        help="disable mixed precision training. for older pytorch versions",
    )
    parser.add_argument("--wandb", action="store_true", help="enable wandb")
    parser.add_argument("--mixup", action="store_true", help="add mixup augumentations")
    parser.add_argument("--net", default="vit")
    parser.add_argument("--bs", type=int, default="512")
    parser.add_argument("--img-size", type=int, default="32")
    parser.add_argument(
        "--weights-from",
        type=str,
        default="",
        help="Path for getting the trained model for resuming training (Should only be used with "
        "--resume)",
    )
    parser.add_argument(
        "--weights-to",
        type=str,
        default="",
        help="Store the trained weights after resuming training session. It will create a new folder "
        "with timestamp in the given path",
    )
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_classes", type=int, default="10")
    parser.add_argument("--num_workers", type=int, default="4")
    parser.add_argument("--n_epochs", type=int, default="200")
    parser.add_argument("--patch", type=int, default="4", help="patch for ViT")
    parser.add_argument("--dimhead", type=int, default="512", help="dimhead for ViT")
    parser.add_argument(
        "--convkernel", type=int, default="8", help="parameter for convmixer"
    )
    parser.add_argument("--watermark", type=str, default="")
    parser.add_argument('--schlr', default='cosine', type=str, help='scheduler')

    args = parser.parse_args()

    if args.wandb:
        if args.watermark == "":
            watermark = "{}_lr{}_{}".format(
                args.net, args.lr, time.strftime("%Y-%m-%d_%H-%M-%S")
            )
        else:
            watermark = args.watermark
        wandb.init(
            project="PyTorch-Cifar-10", entity="iut-hert", name=watermark, config=args
        )
        art = wandb.Artifact(watermark, type="model")
    else:
        art = None

    main(
        args,
        bs=args.bs,
        img_size=args.img_size,
        net=args.net,
        dataset=args.dataset,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
        n_epochs=args.n_epochs,
        patch=args.patch,
        dimhead=args.dimhead,
        convkernel=args.convkernel,
        weights_from=args.weights_from,
        weights_to=args.weights_to,
        resume=args.resume,
        use_amp=not args.noamp,
        aug=args.noaug,
        opt=args.opt,
        artifact=art,
        schlr=args.schlr,
    )
