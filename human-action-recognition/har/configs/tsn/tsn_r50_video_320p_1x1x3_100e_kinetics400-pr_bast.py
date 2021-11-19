# dataset settings
ann_type = 'tanz_base'  # * change accordingly
videos_per_gpu_train = 1 if ann_type == 'tanz_base' else 4
workers_per_gpu_train = 2  # normally it was 4, but it creates a bottleneck in CPU usage
num_classes = 9 if ann_type == 'tanz_base' else 42

# model settings
model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50,
        norm_eval=False),
    cls_head=dict(
        type='TSNHead',
        num_classes=num_classes,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.4,
        init_std=0.01),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips=None))


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
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    dict(type='DecordDecode'),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    # `clip_len=1` represents a `uniform sampling strategy` where all frames are
    # divided into `num_clips` part and one frame is sampled from each part
    #
    # In this case we divide the ~300 frames that we have in a 10s clip
    # into 3 sections with 100 frames and sample 1 frame from each section
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
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
        num_clips=25,
        test_mode=True),
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
evaluation = dict(  # Config of evaluation during training
    interval=5,  # Interval to perform evaluation
    metric_options=dict(
        top_k_accuracy=dict(
        topk=(1, 2, 3, 4, 5))),)
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
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 150

# runtime settings
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ('https://download.openmmlab.com/mmaction/recognition/tsn/'
            'tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb/'
            'tsn_r50_video_320p_1x1x3_100e_kinetics400_rgb_20201014-5ae1ee79.pth')
resume_from = None
workflow = [('train', 1)]
