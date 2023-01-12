# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from functools import partial
from models.vit import ViT
from models.convmixer import ConvMixer
from models.dpn import DPN92, DPN26
from models.crossvit import CrossVisionTransformer
from models.pvt import PyramidVisionTransformer
from models.pvt_v2 import PyramidVisionTransformerV2
from models.visformer import Visformer
from models.resmlp import ResMLP
from models.vip import VisionPermutator, WeightedPermuteMLP
from models.maxvit import MaxViT
from models.mvitv2 import MultiScaleVit

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()

# take in args
usewandb = ~args.nowandb
if usewandb:
    import wandb
    watermark = "{}_lr{}".format(args.net, args.lr)
    wandb.init(project="cifar10-challange",
            name=watermark)
    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = bool(~args.noamp)
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))

# Prepare dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))
elif args.net=='dpn':
    net = DPN26()
elif args.net=='dpn92':
    net = DPN92()
elif args.net=='dpn26':
    net = DPN26()
elif args.net=='crossvit':
    net = CrossVisionTransformer(
    img_size=[240, 224],
    patch_size=[12, 16], 
    embed_dim=[96, 192], 
    depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
    num_heads=[3, 3], 
    mlp_ratio=[4, 4, 1], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    num_classes=10
)
elif args.net=='crossvit_tiny_224':
    net = CrossVisionTransformer(
    img_size=[240, 224],
    patch_size=[12, 16], 
    embed_dim=[96, 192], 
    depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
    num_heads=[3, 3], 
    mlp_ratio=[4, 4, 1], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    num_classes=10
)
elif args.net=='crossvit_small_224':
    net = CrossVisionTransformer(
    img_size=[240, 224],
    patch_size=[12, 16], 
    embed_dim=[192, 384], 
    depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
    num_heads=[6, 6], 
    mlp_ratio=[4, 4, 1], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    num_classes=10
)
elif args.net=='crossvit_base_224':
    net = CrossVisionTransformer(
    img_size=[240, 224],
    patch_size=[12, 16], 
    embed_dim=[384, 768], 
    depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
    num_heads=[12, 12], 
    mlp_ratio=[4, 4, 1], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    num_classes=10
)
elif args.net=='crossvit_9_224':
    net = CrossVisionTransformer(
    img_size=[240, 224],
    patch_size=[12, 16], 
    embed_dim=[128, 256], 
    depth=[[1, 3, 0], [1, 3, 0], [1, 3, 0]],
    num_heads=[4, 4], 
    mlp_ratio=[3, 3, 1], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    num_classes=10
)
elif args.net=='crossvit_15_224':
    net = CrossVisionTransformer(
    img_size=[240, 224],
    patch_size=[12, 16], 
    embed_dim=[192, 384], 
    depth=[[1, 5, 0], [1, 5, 0], [1, 5, 0]],
    num_heads=[6, 6], 
    mlp_ratio=[3, 3, 1], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    num_classes=10
)
elif args.net=='crossvit_18_224':
    net = CrossVisionTransformer(
    img_size=[240, 224],
    patch_size=[12, 16], 
    embed_dim=[224, 448], 
    depth=[[1, 6, 0], [1, 6, 0], [1, 6, 0]],
    num_heads=[7, 7], 
    mlp_ratio=[3, 3, 1], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    num_classes=10
)
elif args.net=='pvt':
    net = PyramidVisionTransformer(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[2, 2, 2, 2], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_tiny':
    net = PyramidVisionTransformer(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[2, 2, 2, 2], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_small':
    net = PyramidVisionTransformer(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 4, 6, 3], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_medium':
    net = PyramidVisionTransformer(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 4, 18, 3], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_large':
    net = PyramidVisionTransformer(
        patch_size=4, 
        embed_dims=[64, 128, 320, 512], 
        num_heads=[1, 2, 5, 8], 
        mlp_ratios=[8, 8, 4, 4], 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        depths=[3, 8, 27, 3], 
        sr_ratios=[8, 4, 2, 1],
        num_classes=10
)
elif args.net=='pvt_huge_v2':
    net = PyramidVisionTransformer(
    patch_size=4, 
    embed_dims=[128, 256, 512, 768], 
    num_heads=[2, 4, 8, 12], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 10, 60, 3], 
    sr_ratios=[8, 4, 2, 1],
    # drop_rate=0.0, drop_path_rate=0.02)
    num_classes=10
)
elif args.net=='pvt_v2':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[32, 64, 160, 256], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[2, 2, 2, 2], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_v2_b0':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[32, 64, 160, 256], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[2, 2, 2, 2], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_v2_b1':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[2, 2, 2, 2], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_v2_b2':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 4, 6, 3], 
    sr_ratios=[8, 4, 2, 1], 
    num_classes=10
)
elif args.net=='pvt_v2_b3':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 4, 18, 3], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_v2_b4':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 8, 27, 3], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_v2_b5':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[4, 4, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 6, 40, 3], 
    sr_ratios=[8, 4, 2, 1],
    num_classes=10
)
elif args.net=='pvt_v2_b2_li':
    net = PyramidVisionTransformerV2(
    patch_size=4, 
    embed_dims=[64, 128, 320, 512], 
    num_heads=[1, 2, 5, 8], 
    mlp_ratios=[8, 8, 4, 4], 
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6), 
    depths=[3, 4, 6, 3], 
    sr_ratios=[8, 4, 2, 1], 
    linear=True, 
    num_classes=10
)
elif args.net=='visformer':
    net = Visformer(
    img_size=224, 
    init_channels=32, 
    embed_dim=384, 
    depth=[7,4,4], 
    num_heads=6, 
    mlp_ratio=4., 
    group=8,
    attn_stage='011', 
    spatial_conv='100', 
    norm_layer=nn.BatchNorm2d, 
    conv_init=True,
    embedding_norm=nn.BatchNorm2d,
    num_classes=10
)
elif args.net=='visformer_tiny':
    net = Visformer(
    img_size=224, 
    init_channels=16, 
    embed_dim=192, 
    depth=[7,4,4], 
    num_heads=3, 
    mlp_ratio=4., 
    group=8,
    attn_stage='011', 
    spatial_conv='100', 
    norm_layer=nn.BatchNorm2d, 
    conv_init=True,
    embedding_norm=nn.BatchNorm2d,
    num_classes=10
)
elif args.net=='visformer_small':
    net = Visformer(
    img_size=224, 
    init_channels=32, 
    embed_dim=384, 
    depth=[7,4,4], 
    num_heads=6, 
    mlp_ratio=4., 
    group=8,
    attn_stage='011', 
    spatial_conv='100', 
    norm_layer=nn.BatchNorm2d, 
    conv_init=True,
    embedding_norm=nn.BatchNorm2d,
    num_classes=10
)
elif args.net=='resmlp':
    net = ResMLP(
    in_channels=3, 
    image_size=224, 
    patch_size=16, 
    num_classes=10,
    dim=384, 
    depth=12, 
    mlp_dim=384*4
)
elif args.net=='vip':
    net = VisionPermutator(
    [4, 3, 8, 3], 
    embed_dims=[384, 384, 384, 384], 
    patch_size=14, 
    transitions=[False, False, False, False],
    segment_dim=[16, 16, 16, 16], 
    mlp_ratios=[3, 3, 3, 3], 
    mlp_fn=WeightedPermuteMLP,
    num_classes=10
)
elif args.net=='vip_s7':
    net = VisionPermutator(
        layers=[4, 3, 8, 3],
        embed_dims=[192, 384, 384, 384],
        patch_size=7,
        transitions=[True, False, False, False],
        segment_dim=[32, 16, 16, 16],
        mlp_ratios=[3, 3, 3, 3],
        mlp_fn=WeightedPermuteMLP,
        num_classes=10
)
elif args.net=='vip_m7':
    net = VisionPermutator(
        layers=[4, 3, 14, 3],
        embed_dims=[256, 256, 512, 512],
        patch_size=7,
        transitions=[False, True, False, False],
        segment_dim=[32, 32, 16, 16],
        mlp_ratios=[3, 3, 3, 3],
        mlp_fn=WeightedPermuteMLP,
        num_classes=10
)
elif args.net=='vip_l7':
    net = VisionPermutator(
        layers=[8, 8, 16, 4],
        embed_dims=[256, 512, 512, 512],
        patch_size=7,
        transitions=[True, False, False, False],
        segment_dim=[32, 16, 16, 16],
        mlp_ratios=[3, 3, 3, 3],
        mlp_fn=WeightedPermuteMLP,
        num_classes=10
)
elif args.net=='maxvit':
    net = MaxViT(
    depths=(2, 2, 5, 2),
    channels=(64, 128, 256, 512),
    embed_dim=64,
    num_classes=10
)
elif args.net=='max_vit_tiny_224':
    net = MaxViT(
    depths=(2, 2, 5, 2),
    channels=(64, 128, 256, 512),
    embed_dim=64,
    num_classes=10
)
elif args.net=='max_vit_small_224':
    net = MaxViT(
    depths=(2, 2, 5, 2),
    channels=(64, 128, 256, 512),
    embed_dim=64,
    num_classes=10
)
elif args.net=='max_vit_base_224':
    net = MaxViT(
    depths=(2, 6, 14, 2),
    channels=(96, 192, 384, 768),
    embed_dim=64,
    num_classes=10
)
elif args.net=='max_vit_large_224':
    net = MaxViT(
    depths=(2, 6, 14, 2),
    channels=(128, 256, 512, 1024),
    embed_dim=128,
    num_classes=10
)
elif args.net=='mvitv2':
    net = MultiScaleVit(
    depths=(1, 2, 5, 2),
    num_classes=10
)
elif args.net=='mvitv2_tiny':
    net = MultiScaleVit(
    depths=(1, 2, 5, 2),
    num_classes=10
)
elif args.net=='mvitv2_small':
    net = MultiScaleVit(
    depths=(1, 2, 11, 2),
    num_classes=10
)
elif args.net=='mvitv2_base':
    net = MultiScaleVit(
    depths=(2, 3, 16, 3),
    num_classes=10
)
elif args.net=='mvitv2_large':
    net = MultiScaleVit(
    depths=(2, 6, 36, 4),
    num_classes=10
)



# For Multi-GPU
if 'cuda' in device:
    print(device)
    print("using data parallel")
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
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
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []

if usewandb:
    wandb.watch(net)
    
net.cuda()
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    if usewandb:
        wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
        "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))
    