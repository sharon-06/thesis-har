_base_ = [
    '../../../_base_/models/slowonly_r50.py',
    '../../../_base_/default_runtime.py'
]

ann_type = 'tanz_base'  # * change accordingly
videos_per_gpu_train = 8 if ann_type == 'tanz_base' else 4
workers_per_gpu_train = 1  # Normally it was 4, but it creates a bottleneck in CPU usage. Check SlowTraining&HighResourceConsumption.md
num_classes = 9 if ann_type == 'tanz_base' else 42

# this is a two stream model (according to the paper)
# slowonly part of slowfast + optical flow from i3d
model = dict(
    backbone=dict(pretrained=None), cls_head=dict(num_classes=num_classes))

# dataset settings
dataset_type = 'VideoDataset'  # can be video | frames | custom
omnisource = False  # The flag indicates using joint training: False because we are fine tuning
data_root = '/mmaction2/'
data_root_val = data_root
data_root_test = ''
ann_file_train = f'data/{ann_type}/tanz_train_list_videos.txt'
ann_file_val = f'data/{ann_type}/tanz_val_list_videos.txt'
ann_file_test = f'/mnt/data_transfer/read/to_process_test/{ann_type}_test_list_videos.txt'

# Pre-processing pipelines
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=8,
        frame_interval=8,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# data config
data = dict(
    videos_per_gpu=videos_per_gpu_train,
    workers_per_gpu=workers_per_gpu_train,
    test_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train_ratio=[2, 1, 1, 1],
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)

# runtime settings
total_epochs = 100
checkpoint_config = dict(interval=5)
evaluation = dict(
    interval=5,
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4, 5))),
)
eval_config = dict(
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4, 5))), )

### runtime settings
# use the pre-trained model on kinetics400
load_from = 'https://download.openmmlab.com/mmaction/recognition/omnisource/slowonly_r50_8x8x1_256e_minikinetics_rgb/omnisource/slowonly_r50_8x8x1_256e_minikinetics_omnisource_rgb_20201030-284cfd3b.pth'
workflow = [('train', 1)]
