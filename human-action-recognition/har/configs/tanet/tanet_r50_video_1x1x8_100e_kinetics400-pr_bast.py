# dataset settings
ann_type = 'tanz_base'  # * change accordingly
videos_per_gpu_train = 1 if ann_type == 'tanz_base' else 4
workers_per_gpu_train = 2  # normally it was 4, but it creates a bottleneck in CPU usage
num_classes = 9 if ann_type == 'tanz_base' else 42

# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='TANet',
        pretrained='torchvision://resnet50',
        depth=50,
        num_segments=8,
        tam_cfg=dict()),
    cls_head=dict(
        type='TSMHead',
        num_classes=num_classes,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))


# dataset settings
dataset_type = 'VideoDataset'
data_root_train = '/home/rejnald/projects/vortanz/mmaction2/'
data_root_val = data_root_train
data_root_test = ''
ann_file_train = f'data/{ann_type}/tanz_train_list_videos.txt'
ann_file_val = f'data/{ann_type}/tanz_val_list_videos.txt'
ann_file_test = f'/mnt/data_transfer/read/to_process_test/{ann_type}_test_list_videos.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    # uniform sampling strategy
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=videos_per_gpu_train,
    workers_per_gpu=workers_per_gpu_train,
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1),
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root_train,
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
# evaluation
evaluation = dict(
    interval=5,
    metrics=['top_k_accuracy', 'mean_class_accuracy'],
    metric_options=dict(topk=(1, 2, 3, 4, 5)))
eval_config = dict(
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4, 5))))

# optimizer
optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=True),
    lr=0.003125,  # office machine
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[50, 75, 90])
total_epochs = 100

# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://download.openmmlab.com/mmaction/recognition/tanet/'
            'tanet_r50_dense_1x1x8_100e_kinetics400_rgb/'
            'tanet_r50_dense_1x1x8_100e_kinetics400_rgb_20210219-032c8e94.pth')
resume_from = None
workflow = [('train', 1)]
