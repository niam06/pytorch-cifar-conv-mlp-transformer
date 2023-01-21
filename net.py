from models import *


def Net(args, img_size, num_classes, bs, patch, dimhead, convkernel):
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
    elif args.net=="mobilenet_v2":
        net=MobileNetV2()
    elif args.net=="convmlp_small":
        net=ConvMLP(blocks=[2, 4, 2],dims=[128, 256, 512],mlp_ratios=[2, 2, 2])
    elif args.net=="convmlp_medium":
        net=ConvMLP(blocks=[3, 6, 3], mlp_ratios=[3, 3, 3], dims=[128, 256, 512])
    elif args.net=="convmlp_large":
        net=ConvMLP(blocks=[4, 8, 3], mlp_ratios=[3, 3, 3], dims=[192, 384, 768])
    elif args.net=="g_mlp":
        net=gMLPForImageClassification()
    elif args.net=="sparse_mlp":
        net=SparseMLP()
    elif args.net=="morph_mlp":
        net=MorphMLP()

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


    return net
