from isegm.utils.exp_imports.default import *
from isegm.data.aligned_augmentation import AlignedAugmentator
MODEL_NAME = 'segformerB3_att_cclvis'
from isegm.data.compose import ComposeDataset,ProportionalComposeDataset
import torch.nn as nn
from isegm.engine.clickatt_trainer_batchconsistant import ISTrainer
from isegm.model.is_refine_segformer_model import SegFormerModel

def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (320, 480)
    model_cfg.num_max_points = 24
    
    refiner_conf = dict(conv_layer='xconv2', mid_dims=96, corr_channel=96)
    
    model = SegFormerModel(model_version = 'b3',
                       use_leaky_relu=True, use_rgb_conv=False, use_disks=True, norm_radius=5, binary_prev_mask=False,
                       with_prev_mask=True, with_aux_output=True, refiner_conf=refiner_conf, only_first_click=True)

    model.to(cfg.device)
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.SEGFORMER_B3)

    return model, model_cfg

def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_refine_loss = WFNL(alpha=0.5, gamma=2) #NormalizedFocalLossSigmoid(alpha=0.5, gamma=2) #WFNL(alpha=0.5, gamma=2)
    loss_cfg.instance_refine_loss_weight = 1.0
    
    '''
    train_augmentator = AlignedAugmentator(ratio=[0.3,1.3], target_size=crop_size, flip=True, 
                                            distribution='Gaussian', gs_center = 0.8, gs_sd = 0.4,
                                            color_augmentator = None
                                    )
    '''
    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.40)),
        HorizontalFlip(),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75),
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = FirstPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2,
                                       use_hierarchy=False,
                                       first_click_center=True)

    trainset_cclvs = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='train',
        augmentator=train_augmentator,
        min_object_area=1000,
        keep_background_prob=0.0,
        points_sampler=points_sampler,
        epoch_len=30000,
        stuff_prob=0.2,
        with_refiner=False
    )

    valset = CocoLvisDataset(
        cfg.LVIS_v1_PATH,
        split='val',
        augmentator=val_augmentator,
        min_object_area=1000,
        points_sampler=points_sampler,
        epoch_len=2000,
        with_refiner=False
    )

    optimizer_params = {
        'lr': 2e-3, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[200, 220], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset_cclvs, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (200, 1)],
                        image_dump_interval=3000,
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=24,
                        sample_prob_sigma=0.8)
    trainer.run(num_epochs=230)