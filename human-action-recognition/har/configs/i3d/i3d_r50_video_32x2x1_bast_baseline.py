_base_ = ['./i3d_r50_32x2x1_100e_kinetics400_rgb.py']

# * this config has explanations on all attributes
# more infos on: https://mmaction2.readthedocs.io/en/latest/tutorials/1_config.html

# dataset settings
ann_type = 'tanz_base' # * _base or _evaluation
videos_per_gpu_train = 8 if ann_type == 'tanz_base' else 4
# workers_per_gpu_train = 4 => created a bottleneck in CPU usage; check SlowTraining&HighResourceConsumption.md
workers_per_gpu_train = 1
# currently utilizing 42 classes for eval annotations
num_classes = 9 if ann_type == 'tanz_base' else 42

## * model settings
model = dict(
    # 2D- The conventional Image Recognition Model
    # 3D- Spatio-temporal model which have proved very useful for videos
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3d', # paper: 3D Residual Networks for Action Recognition
        pretrained2d=True, # using a 2D pre-trained model on Imagenet
        pretrained='torchvision://resnet50', # 3D pre-trained ResNet
        depth=50,
        conv_cfg=dict(type='Conv3d'), # using 3D convolutions
        norm_eval=False, # set BachNormalization layers to eval while training
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)), # inflate dim of each block
        zero_init_residual=False, # whether to init residual blocks with 0
    ),
    cls_head=dict(
        type='I3DHead', # the head varies with the type of architecture
        num_classes=num_classes,
        in_channels=2048,  # the input channels of classification head
        spatial_type='avg',  # type of pooling in spatial dimension
        dropout_ratio=0.5,  # probability in dropout layer
        init_std=0.01),  # std value for linear layer initiation
    # model training and testing settings
    train_cfg=None,  # config for training hyperparameters
    test_cfg=dict(average_clips='prob'))  # config for testing hyperparameters

## * dataset settings
dataset_type = 'VideoDataset'  # can be video | frames | pose | custom
data_root = '/mmaction2/'
data_root_val = data_root
ann_file_train = f'data/{ann_type}/tanz_train_list_videos.txt'
ann_file_val = f'data/{ann_type}/tanz_val_list_videos.txt'
ann_file_test = f'/mnt/data_transfer/read/to_process_test/{ann_type}_test_list_videos.txt'

# config for image normalization used in data pipeline
# https://stats.stackexchange.com/questions/211436/why-normalize-images-by-subtracting-datasets-image-mean-instead-of-the-current
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # mean values of different channels to normalize
    std=[58.395, 57.12, 57.375],  # std values of different channels to normalize
    to_bgr=False, # whether to convert channels from rgb to bgr
)

## * pre-processing pipelines, data augmentation techniques
## For the pre-processing pipelines down below, it might be a good idea to use lazy mode as this will accelerate the training
# each operation takes a dic as input and outputs a dic for the next operation
train_pipeline = [  # list of training pipeline steps
    # The steps DecordInit & DecordDecode are used to decode the video on the fly
    dict(type='DecordInit'), # mmaction/datasets/pipelines/loading.py
    dict(
        type='SampleFrames',
        clip_len=32,  # number of frames sampled for each clip
        frame_interval=1,  # temporal interval of adjacent sampled frames; # frames skipped while sampling
        num_clips=1),
    dict(type='DecordDecode'),
    # augmentations: mmaction/datasets/pipelines/augmentations.py
    dict(type='Resize', scale=(-1, 256)),  # the scale to resize images
    dict(
        type=
        'MultiScaleCrop',  # crop images with a list of randomly selected scales
        input_size=224,
        scales=(1, 0.8),  # width & height scales to be selected
        random_crop=False,
        max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(
        type='Flip',  # flip pipeline
        flip_ratio=0.5),  # probability of implementing flip
    dict(
        type='Normalize',  # normalize pipeline
        **img_norm_cfg),  # config of image normalization
    dict(type='FormatShape', input_format='NCTHW'),  # format final image shape to the given input_format
    dict(
        type=
        'Collect',  # collect pipeline that decides which keys in the data should be passed to the recognizer
        keys=['imgs', 'label'],  # keys of input
        meta_keys=[]),  # meta keys of input
    dict(
        type='ToTensor',  # convert other types to tensor
        keys=['imgs', 'label'])  # keys to be converted from image to tensor
]

val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=1,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type=
        'CenterCrop',  # center crop pipeline, cropping the center area from images
        crop_size=224),
    dict(
        type='Flip',  # flip pipeline
        flip_ratio=0),  # probability of implementig the flip
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# one tweak that could be done here if you get a cuda out of memory error
# could be to change the test pipeline into a light one
# for e.g., num_clips=10 -> num_clips=1
#   dict(type='ThreeCrop', crop_size=256) -> dict(type='CenterCrop', crop_size=224)
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=1,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='ThreeCrop',  # three crop pipeline, cropping 3 areas from images
        crop_size=256),  # the size to crop images
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    # tune these two `_gpu` values according to the Linear Scaling Rule (https://arxiv.org/abs/1706.02677)
    # https://mmaction2.readthedocs.io/en/latest/recognition_models.html
    videos_per_gpu=videos_per_gpu_train,  # originally 8; number of videos in each GPU, i.e. mini-batch size of GPU
    workers_per_gpu=workers_per_gpu_train,  # originally 8; workers (sub-processes) to pre-fetch data for each single gpu; (multithreaded loading for PyTorch)
    test_dataloader=dict(  # Additional config of test dataloader
        videos_per_gpu=1,  # has to be one-one else the mainz gpus won't hold and SIGKILL9 error results
        workers_per_gpu=1),
    val_dataloader=dict(videos_per_gpu=1, workers_per_gpu=1),
    train=dict(  # training dataset config
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        # num_classes=num_classes,
        # multi_class=True,
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        # num_classes=num_classes,
        # multi_class=True,
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix='',
        pipeline=test_pipeline,
        # num_classes=num_classes,
        # multi_class=True,
    )
)

### optimizations
# normally the optimizations goes into a separate file but we want to have a bird's eye view in all the possible
# configs in this particular config file. For the case of i3d, the optimizations can be found at configs/_base_/schedules/sgd_100e.py

# * checkout the linear scaling rule to optimize the learning rate for the #GPUs
optimizer = dict(
    type='SGD',
    lr=0.005, # * linear scaling rule
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40, 80])
total_epochs = 100

### runtime settings
# normally the runtime settings go into a separate file but we want to have a bird's eye view in all the possible
# configs in this particular config file. For the case of i3d, the runtime settings can be found at configs/_base_/default_runtime.py

checkpoint_config = dict(interval=5)
log_config = dict( # config to register logger hook
    interval=20, # interval to print the log
    hooks=[ # hooks to be implemented during training
        dict(type='TextLoggerHook'), # the logger used to record the training process
        # dict(type='TensorboardLoggerHook'), # the tensorboard logger is also supported
    ]
)
evaluation = dict(  # Config of evaluation during training
    interval=5,  # Interval to perform evaluation
    metric_options=dict(
        top_k_accuracy=dict(
        topk=(1, 2, 3, 4, 5))),  # set the top-k accuracy during validation;
    # for training the corresponding head must be modified (i3d_head in this case): https://github.com/open-mmlab/mmaction2/issues/874
    # for testing: eval config below
)
eval_config = dict(
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 4, 5))),)

dist_params = dict(backend='nccl') # parameters to setup distributed training
log_level = 'INFO' # logging level
# load models as a pre-trained model from a given path. Does not resume training
load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth'
resume_from = None # resume training from a particular model
workflow = [('train', 1)] # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once
