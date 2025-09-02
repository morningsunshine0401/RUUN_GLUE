# File: configs/my_aircraft/td-hm_hrnet-w32_myaircraft-512x384.py
# Training command: python tools/train.py configs/my_aircraft/td-hm_hrnet-w32_myaircraft-512x384.py

_base_ = ['/home/runbk0401/mmpose/configs/_base_/default_runtime.py']

# 1. Training schedule - Extended for higher resolution
train_cfg = dict(max_epochs=300, val_interval=10)

# 2. Optimizer with AMP and adjusted learning rate
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='Adam', lr=2.5e-4),  # Reduced from 5e-4 due to smaller batch size
    accumulative_counts=8  # Increased to maintain effective batch size
)

# 3. Learning-rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=1000,  # Longer warmup for higher resolution
        start_factor=0.001,
        by_epoch=False,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        milestones=[240, 280],  # Adjusted for longer training
        gamma=0.1,
        by_epoch=True,
    ),
]

# Automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=128)

# 4. Checkpoint hook: save best by AP
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater')
)

# 5. Heatmap codec for 512x384 resolution
codec = dict(
    type='MSRAHeatmap',
    input_size=(384, 512),   # Height x Width: 384x512
    heatmap_size=(96, 128),  # 1:4 ratio: 96x128
    sigma=2
)

# 6. Model definition: HRNet-W32 backbone + heatmap head
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)
            )
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'
        )
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=6,             # Your dataset has 6 keypoints
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ),
)

# 7. Dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
#data_root = '/home/runbk0401/mmpose/data/aircraft_coco_dataset_2'
data_root = '/home/runbk0401/mmpose/data/20250705_aircraft_coco'

# 8. Updated data pipelines for 512x384
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# 9. Dataloaders with reduced batch sizes for higher resolution
train_dataloader = dict(
    batch_size=16,  # Reduced from 64 due to higher resolution memory requirements
    num_workers=4,
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
    )
)

val_dataloader = dict(
    batch_size=8,  # Reduced from 32
    num_workers=4,
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
    )
)

test_dataloader = val_dataloader

# 10. Evaluator: use standard COCO keypoint metric (OKS/AP)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/val3.json',
)
test_evaluator = val_evaluator

# Work directory
work_dir = './work_dirs/td-hm_hrnet-w32_myaircraft-512x384_20250705'