# # /home/runbk0401/miniconda3/envs/yolov8_env/bin/python tools/train.py configs/my_aircraft/td-hm_hrnet-w32_myaircraft-256x192.py

# #(export PATH="/home/runbk0401/miniconda3/envs/openmmlab/bin:$PATH" 이후)   python tools/train.py configs/my_aircraft/td-hm_hrnet-w32_myaircraft-256x192.py


# # File: configs/my_aircraft/td-hm_hrnet-w32_myaircraft-256x192.py



# /home/runbk0401/miniconda3/envs/yolov8_env/bin/python tools/train.py configs/my_aircraft/td-hm_hrnet-w32_myaircraft-256x192.py

# File: configs/my_aircraft/td-hm_hrnet-w32_myaircraft-256x192.py

_base_ = ['/home/runbk0401/mmpose/configs/_base_/default_runtime.py']  # inherit default runtime & schedule

# 1. Training schedule
train_cfg = dict(max_epochs=210, val_interval=10)

# # 2. Optimizer (Adam)
# optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4),
#                      accumulative_counts=4 ####
#                      )

# Enable mixed precision to halve memory usage
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(type='Adam', lr=5e-4),
    accumulative_counts=4  # Optional: for gradient accumulation
)

# 3. Learning-rate schedule
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=128) #512

# 4. Checkpoint hook: save best by AP
default_hooks = dict(
    checkpoint=dict(save_best='coco/AP', rule='greater')
)

# 5. Heatmap codec
codec = dict(
    type='MSRAHeatmap',
    input_size=(192, 256),   # HRNet uses H=256, W=192 by default
    heatmap_size=(48, 64),
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
        out_channels=6,             # YOUR dataset has 6 keypoints
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
data_root = '/home/runbk0401/mmpose/data/aircraft_coco_dataset'  # <<— replace with your real path

# 8. Updated data pipelines (MMPose v1.x API)
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

# 9. Dataloaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,     # adjust to your machine
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/train.json',     # ← your train.json (must exist)
        data_prefix=dict(img='images/train/'),  # ← train images folder
        pipeline=train_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/my_aircraft.py')
    )
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/val.json',       # ← the file you just uploaded
        data_prefix=dict(img='images/val/'),    # ← val images folder
        test_mode=True,
        pipeline=val_pipeline,
        metainfo=dict(from_file='configs/_base_/datasets/my_aircraft.py')
    )
)

test_dataloader = val_dataloader  # reuse val for testing if no separate test split

# 10. Evaluator: use standard COCO keypoint metric (OKS/AP)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + '/annotations/val.json',
    #metric='keypoints',
)
test_evaluator = val_evaluator

# Work directory
work_dir = './work_dirs/td-hm_hrnet-w32_myaircraft-256x192'



# _base_ = ['/home/runbk0401/mmpose/configs/_base_/default_runtime.py']  # inherit default runtime & schedule


# custom_imports = dict(
#     imports=[
#         'mmpose.models',                       # registers TopdownPoseEstimator, HeatmapHead, etc.
#         'mmpose.datasets',
#         'mmpose.datasets.transforms.topdown_transforms',                     # registers CocoDataset, etc.
#         #'mmpose.core',                         # registers evaluators, hooks, losses, etc.
#         'mmpose.visualization.local_visualizer'  # registers LocalVisualizer if you need it
#     ],
#     allow_failed_imports=False)

# # 1. Training schedule
# train_cfg = dict(max_epochs=210, val_interval=10)

# # 2. Optimizer (Adam)
# optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))

# # 3. Learning-rate schedule
# param_scheduler = [
#     dict(
#         type='LinearLR',
#         begin=0,
#         end=500,
#         start_factor=0.001,
#         by_epoch=False,
#     ),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=210,
#         milestones=[170, 200],
#         gamma=0.1,
#         by_epoch=True,
#     ),
# ]
# auto_scale_lr = dict(base_batch_size=512)

# # 4. Checkpoint hook: save best by AP
# default_hooks = dict(
#     checkpoint=dict(save_best='coco/AP', rule='greater')
# )

# # 5. Heatmap codec
# codec = dict(
#     type='MSRAHeatmap',
#     input_size=(192, 256),   # HRNet uses H=256, W=192 by default
#     heatmap_size=(48, 64),
#     sigma=2
# )

# # 6. Model definition: HRNet-W32 backbone + heatmap head
# model = dict(
#     type='TopdownPoseEstimator',
#     data_preprocessor=dict(
#         type='PoseDataPreprocessor',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         bgr_to_rgb=True
#     ),
#     backbone=dict(
#         type='HRNet',
#         in_channels=3,
#         extra=dict(
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(4,),
#                 num_channels=(64,)
#             ),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 block='BASIC',
#                 num_blocks=(4, 4),
#                 num_channels=(32, 64)
#             ),
#             stage3=dict(
#                 num_modules=4,
#                 num_branches=3,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4),
#                 num_channels=(32, 64, 128)
#             ),
#             stage4=dict(
#                 num_modules=3,
#                 num_branches=4,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4, 4),
#                 num_channels=(32, 64, 128, 256)
#             )
#         ),
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'
#         )
#     ),
#     head=dict(
#         type='HeatmapHead',
#         in_channels=32,
#         out_channels=6,             # YOUR dataset has 6 keypoints
#         deconv_out_channels=None,
#         loss=dict(type='KeypointMSELoss', use_target_weight=True),
#         decoder=codec
#     ),
#     test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True),
# )

# # 7. Dataset settings
# dataset_type = 'CocoDataset'
# data_mode = 'topdown'
# data_root = '/home/runbk0401/mmpose/data/aircraft_coco_dataset'  # <<— replace with your real path

# # 8. Data pipelines
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     #dict(type='TopDownGetBBoxCenterScale', padding=1.25),
#     #dict(type='TopDownRandomFlip', flip_prob=0.5),
#     #dict(type='TopDownRandomHalfBody'),
    
#     #dict(type='GetBBoxCenterScale'),
#     #dict(type='RandomFlip', direction='horizontal'),
#     #dict(type='RandomHalfBody'),

#     #dict(type='TopDownRandomBBoxTransform'),
#     dict(type='RandomBBoxTransform'),

#     dict(type='TopdownAffine', input_size=codec['input_size']),
#     dict(type='ToTensor'),
#     dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     dict(
#         type='TopDownGenerateTarget',
#         encoding='MSRA',
#         sigma=2,
#         unbiased_encoding=True
#     ),
#     dict(type='Collect', keys=['img', 'target', 'target_weight'], meta_keys=['image_file', 'joints_3d', 'joints_3d_visible'])
# ]

# val_pipeline = [
#     dict(type='LoadImageFromFile'),
#     #dict(type='TopDownGetBBoxCenterScale', padding=1.25),
#     dict(type='TopdownAffine', input_size=codec['input_size']),
#     dict(type='ToTensor'),
#     dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     dict(type='Collect', keys=['img'], meta_keys=['image_file', 'joints_3d', 'joints_3d_visible'])
# ]

# # 9. Dataloaders
# train_dataloader = dict(
#     batch_size=64,
#     num_workers=4,     # adjust to your machine
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_mode=data_mode,
#         ann_file='annotations/train.json',     # ← your train.json (must exist)
#         data_prefix=dict(img='images/train/'),  # ← train images folder
#         pipeline=train_pipeline,
#         metainfo=dict(from_file='configs/_base_/datasets/my_aircraft.py')
#     )
# )

# val_dataloader = dict(
#     batch_size=32,
#     num_workers=4,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_mode=data_mode,
#         ann_file='annotations/val.json',       # ← the file you just uploaded
#         data_prefix=dict(img='images/val/'),    # ← val images folder
#         test_mode=True,
#         pipeline=val_pipeline,
#         metainfo=dict(from_file='configs/_base_/datasets/my_aircraft.py')
#     )
# )

# test_dataloader = val_dataloader  # reuse val for testing if no separate test split

# # 10. Evaluator: use standard COCO keypoint metric (OKS/AP)
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file=data_root + '/annotations/val.json',
#     metric='keypoints',
# )
# test_evaluator = val_evaluator

# # ───────── Override any visualization inherited from default_runtime.py ─────────
# vis_backends = None
# visualizer   = None

# work_dir = './work_dirs/td-hm_hrnet-w32_myaircraft-256x192'
