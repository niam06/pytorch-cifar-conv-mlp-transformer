import torch
import torch.nn as nn
import torch.optim as optim
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
import test


def main(args,
         bs=512,
         img_size=32,
         resume=False,
         n_epochs=200,
         patch=4,
         dimhead=512,
         convkernel=8,
         num_classes=10,
         num_workers=4,
         dataset='cifar10',
         weights_from='',
         weights_to='',
         net='res18',
         use_amp=False,
         aug=True,
         opt='adam',
         schlr='cosine',
         artifact=None,):

    usewandb = args.wandb
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # need to implement resume

    print('==> Preparing data..')

    if net in ['cait', 'simplevit', 'vit_small', 'vit_tiny', 'cait_small', 'swin_small', 'swin_base', 'swin_large', 'swin_tiny', 'gcvit_tiny', 'gcit_small', 'gcit_base', 'gcit_large']:
        img_size = 224
    elif net in ['swinmlp_tiny_c6', 'swinmlp_tiny_c12', 'swinmlp_tiny_c24']:
        img_size = 256

    if dataset == 'cifar10':
        num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        # Add RandAugment with N, M(hyperparameter)
        if aug:
            N = 2
            M = 14
            transform_train.transforms.insert(0, RandAugment(N, M))

        # Prepare dataset
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    elif dataset == 'cifar100':
        num_classes = 100
        # train-test sampler need to be implemented

    elif dataset == 'imagenet':
        num_classes = 1000
        # train-test sampler need to be implemented

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs, shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=bs, shuffle=False, num_workers=num_workers)

    # Model factory..
    print('==> Building model..')

    if args.net == 'res18':
        net = ResNet18()
    elif args.net == 'vgg':
        net = VGG('VGG19')
    elif args.net == 'res34':
        net = ResNet34()
    elif args.net == 'res50':
        net = ResNet50()
    elif args.net == 'res101':
        net = ResNet101()
    elif args.net == 'resnext':
        net = ResNeXt29_2x64d()
    elif args.net == "senet":
        net = SENet18()
    elif args.net=='densenet121':
        net = DenseNet121()
    elif args.net=='densenet161':
        net = DenseNet161()
    elif args.net=='densenet169':
        net = DenseNet169()
    elif args.net=='densenet201':
        net = DenseNet201()
    elif args.net=='densenet':
        net = densenet_cifar()
    elif args.net=="googlenet":
        net = GoogLeNet()
    elif args.net=="efficientnetb0":
        net = EfficientNetB0()
    elif args.net=="dla":
        net = DLA()
    elif args.net=='convnext_tiny':
        net = convnext_tiny()
    elif args.net=='convnext_small':
        net = convnext_small()
    elif args.net=='convnext_base':
        net = convnext_base()
    elif args.net=='convnext_large':
        net = convnext_large()
    elif args.net=='convnext_xlarge':
        net = convnext_xlarge()
    elif args.net=='raft_mlp':
        net = RaftMLP(layers=[
                {"depth": 12,
                "dim": 768,
                "patch_size": 16,
                "raft_size": 4}],
            gap = True)
    elif args.net=='regnetx200':
        net = RegNetX_200MF()
    elif args.net=='regnetx400':
        net = RegNetX_400MF()
    elif args.net=="regnety400":
        net = RegNetY_400MF()
    elif args.net=="xception":
        net = xception()
    elif args.net == "convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=convkernel,
                        patch_size=1, n_classes=num_classes)
    elif args.net=="s2mlpv1_deep":
      net = S2MLPv1_deep()
    elif args.net=="s2mlpv2_wide":
      net = S2MLPv1_wide()
    elif args.net == "mlpmixer":
        net = MLPMixer(
            image_size=img_size,
            channels=3,
            patch_size=patch,
            dim=dimhead,
            depth=6,
            num_classes=num_classes
        )
    elif args.net == "vit_small":
        net = ViT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "vit_tiny":
        net = ViT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=4,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )
    elif args.net == "simplevit":
        net = SimpleViT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,
            heads=8,
            mlp_dim=512
        )
    elif args.net == "cait":
        net = CaiT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,   # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )
    elif args.net == "cait_small":
        net = CaiT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,   # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05
        )
    elif args.net == "swin_tiny":
        net = swin_t(num_classes=num_classes)
    elif args.net == "swin_small":
        net = swin_s(num_classes=num_classes)
    elif args.net == "swin_base":
        net = swin_b(num_classes=num_classes)
    elif args.net == "swin_large":
        net = swin_l(num_classes=num_classes)
    elif args.net == "hrnet_18v1":
        net = hrnet_18v1()
        opt = 'sgd'
    elif args.net == "hrnet_w32":
        net = hrnet_w32()
        opt = 'sgd'
    elif args.net == "hrnet_w64":
        net = hrnet_w64()
        opt = 'sgd'
    elif args.net == "squeezenet":
        net = SqueezeNet(bs)
    elif args.net == "gcvit_tiny":
        net = gc_vit_tiny(num_classes=num_classes)
    elif args.net == "gcvit_small":
        net = gc_vit_small(num_classes=num_classes)
    elif args.net == "gcvit_base":
        net = gc_vit_base_384(num_classes=num_classes)
    elif args.net == "gcvit_large":
        net = gc_vit_large_384(num_classes=num_classes)
    elif args.net == "swinmlp_base":
        net = SwinMLP(drop_path_rate=0.5, embed_dim=128, depths=[
                      2, 2, 18, 2], num_heads=[4, 8, 16, 32], num_classes=num_classes)
    elif args.net == "swinmlp_tiny_c6":
        net = SwinMLP(drop_path_rate=0.2, img_size=img_size, embed_dim=96, depths=[
                      2, 2, 6, 2], num_heads=[16, 32, 64, 128], window_size=8, num_classes=num_classes)
    elif args.net == "swinmlp_tiny_c12":
        net = SwinMLP(drop_path_rate=0.2, img_size=img_size, embed_dim=96, depths=[
                      2, 2, 6, 2], num_heads=[8, 16, 32, 64], window_size=8, num_classes=num_classes)
    elif args.net == "swinmlp_tiny_c24":
        net = SwinMLP(drop_path_rate=0.2, img_size=img_size, embed_dim=96, depths=[
                      2, 2, 6, 2], num_heads=[4, 8, 16, 32], window_size=8, num_classes=num_classes)
    else:
        print("No model found")
        exit()

    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        print("using data parallel")
        net = torch.nn.DataParallel(net)  # make parallel
        cudnn.benchmark = True

    if opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(
            'checkpoint'), 'Error: no checkpoint directory found!'
        if weights_from != '':
            checkpoint = torch.load(weights_from)
        else:
            try:
                artifact = wandb.use_artifact(
                    'iut-hert/PyTorch-Cifar-10/run_' + args.net + ':latest', type='model')
                artifact_dir = artifact.download()
                checkpoint = torch.load(
                    artifact_dir+"/wandb_best_{}.pt".format(args.net))
            except:
                print("No checkpoint found")
                wandb.finish()
                exit()
        net.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        if checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer loaded")

        del checkpoint  # current, saved

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    # use cosine scheduling
    if schlr == "cosine":
         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=200)
    if schlr == "reduceonplateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')

    # Training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    list_loss = []
    list_acc = []

    if usewandb:
        wandb.watch(net)

    start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = 'output/' + 'rerun_' + args.net + "/" + start_time + "/"

    if not os.path.exists(path):
        os.makedirs(path)

    net.cuda()
    for epoch in range(start_epoch, n_epochs):
        start = time.time()
        print('\nEpoch: %d' % epoch)
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

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        trainloss = train_loss/(batch_idx+1)

        val_loss, acc, best_acc = test.test(
            epoch, net, testloader, device, criterion, optimizer, scaler, best_acc, args)

        list_loss.append(val_loss)
        list_acc.append(acc)

        # Log training..
        if usewandb:
            wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "train_acc" : 100.*correct/total, "lr": optimizer.param_groups[0]["lr"],
                       "epoch_time": time.time()-start})

        # Write out csv..
        with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss)
            writer.writerow(list_acc)
        print(list_loss)

        checkpoint = {'epoch': epoch,
                      'model': net.state_dict(),
                      'optimizer': optimizer.state_dict()}

        torch.save(checkpoint, path + 'latest.pt')

        if acc == best_acc:
            checkpoint['optimizer'] = None
            torch.save(checkpoint, path + 'best.pt')
        
        scheduler.step()  # step cosine scheduling

    # writeout wandb
    if usewandb:
        model_artifact = wandb.Artifact('re_run_' + args.net,
                                        type='model',
                                        metadata={
                                            'original_url': str(path),
                                            'epochs_trained': epoch + 1,
                                            'total_epochs': args.n_epochs,
                                            'best_acc': best_acc})
        # model_artifact.add_file(path + 'latest.pt', name="wandb_latest_{}_lr{}.pt".format(args.net, args.lr))
        model_artifact.add_file(
            path + 'best.pt', name="wandb_best_{}.pt".format(args.net))
        wandb.log_artifact(model_artifact)
        wandb.finish()


if __name__ == '__main__':
    # parsers
    parser = argparse.ArgumentParser(description='PyTorch CV Models Training')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')  # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--opt', default='adam', type=str, help='optimizer')
    parser.add_argument('--schlr', default='cosine', type=str, help='scheduler')
    parser.add_argument('--noaug', action='store_true',
                        help='disable use randomaug')
    parser.add_argument('--noamp', action='store_true',
                        help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--wandb', action='store_true', help='enable wandb')
    parser.add_argument('--mixup', action='store_true',
                        help='add mixup augumentations')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--bs', type=int, default='512')
    parser.add_argument('--img-size', type=int, default="32")
    parser.add_argument('--weights-from', type=str, default='',
                        help='Path for getting the trained model for resuming training (Should only be used with '
                             '--resume)')
    parser.add_argument('--weights-to', type=str, default='',
                        help='Store the trained weights after resuming training session. It will create a new folder '
                             'with timestamp in the given path')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default='10')
    parser.add_argument('--num_workers', type=int, default='4')
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', type=int, default='4', help="patch for ViT")
    parser.add_argument('--dimhead', type=int,
                        default="512", help="dimhead for ViT")
    parser.add_argument('--convkernel', type=int,
                        default='8', help="parameter for convmixer")
    parser.add_argument('--watermark', type=str, default='')

    args = parser.parse_args()

    if args.wandb:
        if args.watermark == '':
            watermark = "{}_lr{}_{}".format(
                args.net, args.lr, time.strftime("%Y-%m-%d_%H-%M-%S"))
        else:
            watermark = args.watermark
        wandb.init(project='PyTorch-Cifar-10', entity="iut-hert",
                   name=watermark, config=args)
        art = wandb.Artifact(watermark, type="model")
    else:
        art =None

    main(args,
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
         artifact=art,)
