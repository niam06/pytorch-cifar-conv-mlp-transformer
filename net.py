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
        net = ConvMixer(
            256, 16, kernel_size=convkernel, patch_size=1, n_classes=num_classes
        )
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
    elif args.net=='cyclemlp':
        net = CycleNet(
        transitions = [True, True, True, True],
        layers = [2, 2, 4, 2],
        mlp_ratios = [4, 4, 4, 4],
        embed_dims = [64, 128, 320, 512],
        num_classes=10
    )
    elif args.net=='asmlp':
        net = AS_MLP(
        num_classes=10
    )
    elif args.net == 'gfnet':
            net = GFNet( 
            patch_size=16, 
            embed_dim=256, 
            depth=12, 
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=10
    )
    elif args.net == 'gfnet_ti':
            net = GFNet( 
            patch_size=16, 
            embed_dim=256, 
            depth=12, 
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=10
    )
    elif args.net == 'gfnet_s':
            net = GFNet(
            patch_size=16, 
            embed_dim=384, 
            depth=19, 
            mlp_ratio=4, 
            drop_path_rate=0.15,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=10
    )
    elif args.net == 'gfnet_b':
            net = GFNet(
            patch_size=16, 
            embed_dim=512, 
            depth=19, 
            mlp_ratio=4, 
            drop_path_rate=0.25,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=10
    )
    elif args.net == 'gfnet_h_ti':
            net = GFNetPyramid(
            patch_size=4, 
            embed_dim=[64, 128, 256, 512], 
            depth=[3, 3, 10, 3],
            mlp_ratio=[4, 4, 4, 4],
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            drop_path_rate=0.1,
            num_classes=10
    )
    elif args.net == 'msmlp':
            net = MS_MLP(
            embed_dim=96,
            depths=[ 3, 3, 9, 3 ],
            patch_size=4,
            num_classes=10
    )
    elif args.net == 'msmlp_t':
            net = MS_MLP(
            embed_dim=96,
            depths=[ 3, 3, 9, 3 ],
            patch_size=4,
            num_classes=10
    )
    elif args.net == 'msmlp_s':
            net = MS_MLP(
            embed_dim=96,
            depths=[ 3, 3, 27, 3 ],
            patch_size=4,
            num_classes=10
    )
    elif args.net == 'msmlp_b':
            net = MS_MLP(
            embed_dim=128,
            depths=[ 3, 3, 27, 3 ],
            patch_size=4,
            num_classes=10
    )
    elif args.net == 'wavemlp':
            net = WaveNet(
            transitions = [True, True, True, True],
            layers = [2, 2, 4, 2],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [64, 128, 320, 512],
            patch_size=7,
            num_classes=10
    )
    elif args.net == 'wavemlp_t':
            net = WaveNet(
            transitions = [True, True, True, True],
            layers = [2, 2, 4, 2],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [64, 128, 320, 512],
            patch_size=7,
            num_classes=10
    )
    elif args.net == 'wavemlp_s':
            net = WaveNet(
            transitions = [True, True, True, True],
            layers = [2, 3, 10, 3],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [64, 128, 320, 512],
            patch_size=7,
            num_classes=10
    )
    elif args.net == 'wavemlp_m':
            net = WaveNet(
            transitions = [True, True, True, True],
            layers = [3, 4, 18, 3],
            mlp_ratios = [8, 8, 4, 4],
            embed_dims = [64, 128, 320, 512],
            patch_size=7,
            num_classes=10
    )
    elif args.net == 'wavemlp_b':
            net = WaveNet(
            transitions = [True, True, True, True],
            layers = [2, 2, 8, 2],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [96, 192, 384, 768],
            patch_size=7,
            num_classes=10
    )
    elif args.net == 'hiremlp':
            net = HireMLPNet(
            layers = [2, 2, 4, 2],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [64, 128, 320, 512],
            pixel = [4, 3, 3, 2],
            step_stride = [2, 2, 3, 2],
            step_dilation = [2, 2, 1, 1],
            step_pad_mode = 'c',
            pixel_pad_mode = 'c',
            num_classes=10
    )
    elif args.net == 'hiremlp_t':
            net = HireMLPNet(
            layers = [2, 2, 4, 2],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [64, 128, 320, 512],
            pixel = [4, 3, 3, 2],
            step_stride = [2, 2, 3, 2],
            step_dilation = [2, 2, 1, 1],
            step_pad_mode = 'c',
            pixel_pad_mode = 'c',
            num_classes=10
    )
    elif args.net == 'hiremlp_s':
            net = HireMLPNet(
            layers = [3, 4, 10, 3],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [64, 128, 320, 512],
            pixel = [4, 3, 3, 2],
            step_stride = [2, 2, 3, 2],
            step_dilation = [2, 2, 1, 1],
            step_pad_mode = 'c',
            pixel_pad_mode = 'c',
            num_classes=10
    )
    elif args.net == 'hiremlp_b':
            net = HireMLPNet(
            layers = [4, 6, 24, 3],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [64, 128, 320, 512],
            pixel = [4, 3, 3, 2],
            step_stride = [2, 2, 3, 2],
            step_dilation = [2, 2, 1, 1],
            step_pad_mode = 'c',
            pixel_pad_mode = 'c',
            num_classes=10
    )
    elif args.net == 'hiremlp_l':
            net = HireMLPNet(
            layers = [4, 6, 24, 3],
            mlp_ratios = [4, 4, 4, 4],
            embed_dims = [96, 192, 384, 768],
            pixel = [4, 3, 3, 2],
            step_stride = [2, 2, 3, 2],
            step_dilation = [2, 2, 1, 1],
            step_pad_mode = 'c',
            pixel_pad_mode = 'c',
            num_classes=10
    )
    
    else:
        print("No model found")
        exit()

    return net