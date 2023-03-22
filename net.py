from models import *


def Net(args, img_size, num_classes, bs, patch, dimhead, convkernel):
    """_summary_



    Parameters
    ----------
    args : _type_ _description_
    img_size : _type_ _description_
    num_classes : _type_ _description_
    bs : _type_ _description_
    patch : _type_ _description_
    dimhead : _type_ _description_
    convkernel : _type_ _description_

    Returns
    -------
    _type_ _description_
    """
    if args.net == "res18":
        net = ResNet18()
    elif args.net == "vgg":
        net = VGG("VGG19")
    elif args.net == "res34":
        net = ResNet34()
    elif args.net == "res50":
        net = ResNet50()
    elif args.net == "res101":
        net = ResNet101()
    elif args.net == "resnext":
        net = ResNeXt29_2x64d()
    elif args.net == "senet":
        net = SENet18()
    elif args.net == "convmixer":
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(
            256, 16, kernel_size=convkernel, patch_size=1, n_classes=num_classes
        )
    elif args.net == "mlpmixer":
        net = MLPMixer(
            image_size=img_size,
            channels=3,
            patch_size=patch,
            dim=dimhead,
            depth=6,
            num_classes=num_classes,
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
            emb_dropout=0.1,
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
            emb_dropout=0.1,
        )
    elif args.net == "simplevit":
        net = SimpleViT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,
            heads=8,
            mlp_dim=512,
        )
    elif args.net == "cait":
        net = CaiT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,  # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=8,
            mlp_dim=512,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05,
        )
    elif args.net == "cait_small":
        net = CaiT(
            image_size=img_size,
            patch_size=patch,
            num_classes=num_classes,
            dim=dimhead,
            depth=6,  # depth of transformer for patch to patch attention only
            cls_depth=2,  # depth of cross attention of CLS tokens to patch
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1,
            layer_dropout=0.05,
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
        opt = "sgd"
    elif args.net == "hrnet_w32":
        net = hrnet_w32()
        opt = "sgd"
    elif args.net == "hrnet_w64":
        net = hrnet_w64()
        opt = "sgd"
    elif args.net == "squeezenet":
        net = SqueezeNet()
    elif args.net == "gcvit_tiny":
        net = gc_vit_tiny(num_classes=num_classes)
    elif args.net == "gcvit_small":
        net = gc_vit_small(num_classes=num_classes)
    elif args.net == "gcvit_base":
        net = gc_vit_base_384(num_classes=num_classes)
    elif args.net == "gcvit_large":
        net = gc_vit_large_384(num_classes=num_classes)
    elif args.net == "swinmlp_base":
        net = SwinMLP(
            drop_path_rate=0.5,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            num_classes=num_classes,
        )
    elif args.net == "swinmlp_tiny_c6":
        net = SwinMLP(
            drop_path_rate=0.2,
            img_size=img_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[16, 32, 64, 128],
            window_size=8,
            num_classes=num_classes,
        )
    elif args.net == "swinmlp_tiny_c12":
        net = SwinMLP(
            drop_path_rate=0.2,
            img_size=img_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[8, 16, 32, 64],
            window_size=8,
            num_classes=num_classes,
        )
    elif args.net == "swinmlp_tiny_c24":
        net = SwinMLP(
            drop_path_rate=0.2,
            img_size=img_size,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=8,
            num_classes=num_classes,
        )
    else:
        print("No model found")
        exit()

    return net
