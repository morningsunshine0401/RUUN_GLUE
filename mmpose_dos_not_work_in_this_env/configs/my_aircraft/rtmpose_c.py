# Optimized RTMPose-L Aircraft Configuration - 384x288
# File: configs/my_aircraft/rtmpose-l_aircraft-384x288-optimized.py

_base_ = ['/home/runbk0401/mmpose/configs/_base_/default_runtime.py']

# IMPROVED TRAINING PARAMETERS FOR SMALL DATASET
max_epochs = 600 #400  # Increased from 270 - small datasets need more epochs
base_lr = 5e-4 #1e-3    # Reduced from 4e-3 - much more conservative for small dataset

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# IMPROVED OPTIMIZER - Lower weight decay for small dataset
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=base_lr, 
        weight_decay=0.01),  # Reduced from 0.05 to prevent overfitting
    # paramwise_cfg=dict(
    #     norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
    clip_grad=dict(max_norm=1.0, norm_type=2)  # ← Add this line)
    )

# IMPROVED LEARNING RATE SCHEDULE - More conservative for small dataset
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=2000),  # Longer warmup: 2000 vs 1000 iterations
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.01,  # Lower minimum: 0.01 vs 0.05
        begin=200,  # Start cosine annealing later: epoch 200 vs 135
        end=max_epochs,
        T_max=max_epochs - 200,  # Adjust T_max accordingly
        by_epoch=True,
        convert_to_iter_based=True),
]

# IMPROVED AUTO SCALING - More appropriate for smaller batch sizes
auto_scale_lr = dict(base_batch_size=128)  # Reduced from 512

# codec settings - same as before
codec = dict(
    type='SimCCLabel',
    input_size=(288, 384),
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings - same as before but with dropout for regularization
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=1024,
        out_channels=6,  # Your 6 aircraft keypoints
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.1,  # Added dropout: 0.1 vs 0.0 for regularization
            drop_path=0.1,     # Added drop path: 0.1 vs 0.0
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    #test_cfg=dict(flip_test=True))
    test_cfg=dict(flip_test=False))

# dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
#data_root = '/home/runbk0401/mmpose/data/aircraft_coco_dataset_3'
#data_root = '/home/runbk0401/mmpose/data/20250705_aircraft_coco_rtm'
#data_root = '/home/runbk0401/mmpose/data/20250724'

data_root = '/home/runbk0401/mmpose/data/20250829'

backend_args = dict(backend='local')

# IMPROVED TRAINING PIPELINE - More aggressive augmentation for small dataset
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', 
        scale_factor=[0.5, 1.5],    # More aggressive scaling: [0.5,1.5] vs [0.6,1.4]
        rotate_factor=90),          # More rotation: 90° vs 80°
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.2),           # Increased probability: 0.2 vs 0.1
            dict(type='MedianBlur', p=0.2),     # Increased probability: 0.2 vs 0.1
            dict(type='GaussNoise', p=0.1),     # Added noise augmentation
            dict(
                type='CoarseDropout',
                max_holes=2,          # More holes: 2 vs 1
                max_height=0.3,       # Smaller holes: 0.3 vs 0.4
                max_width=0.3,        # Smaller holes: 0.3 vs 0.4
                min_holes=1,
                min_height=0.1,       # Smaller min: 0.1 vs 0.2
                min_width=0.1,        # Smaller min: 0.1 vs 0.2
                p=0.8),               # Lower probability: 0.8 vs 1.0
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# train_pipeline = [
#     dict(type='LoadImage', backend_args=backend_args),
#     dict(type='GetBBoxCenterScale'),
#     dict(type='TopdownAffine', input_size=codec['input_size']),
#     dict(type='GenerateTarget', encoder=codec),
#     dict(type='PackPoseInputs'),
# ]

# CONSERVATIVE VALIDATION PIPELINE
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# ADJUSTED DATA LOADERS - Smaller batch sizes for stability
train_dataloader = dict(
    batch_size=8,#16,  # Reduced from 32 for more stable gradients
    num_workers=2,#4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train3.json',
        data_prefix=dict(img='images/train/'),
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/my_aircraft.py')
    ))

val_dataloader = dict(
    batch_size=8,#16,  # Reduced from 32
    num_workers=2,#4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val3.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/my_aircraft.py')
    ))

test_dataloader = val_dataloader

# IMPROVED HOOKS - Save more frequently and use EMA
default_hooks = dict(
    checkpoint=dict(
        save_best='coco/AP', 
        rule='greater', 
        max_keep_ckpts=3,      # Keep more checkpoints: 3 vs 1
        save_last=True,        # Always save last checkpoint
        interval=10))          # Save every 10 epochs

# EMA and early stopping for better training
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,       # More conservative EMA: 0.0001 vs 0.0002
        update_buffers=True,
        priority=49),
    # Optional: Early stopping if you want to prevent overfitting
    # dict(
    #     type='EarlyStoppingHook',
    #     monitor='coco/AP',
    #     patience=50,
    #     rule='greater')
]




# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/val3.json')
test_evaluator = val_evaluator

# # visualizer
visualizer = dict(
    _delete_=True,
    type='PoseVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

# visualizer = None



# work directory
#work_dir = './work_dirs/rtmpose-l_aircraft-384x288_20250808'

work_dir = './work_dirs/rtmpose-l_aircraft-384x288_20250901'
