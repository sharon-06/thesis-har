_base_ = ['./i3d_r50_video_32x2x1_100e_kinetics400_rgb.py']

### Here, for our purposes the weights of the pretrained model are used apart from the final prediction layer

### model settings
model = dict(  # config of a model
    type='Recognizer3D',  # type of i3d??
    backbone=dict(  # dict for backbone
        type='ResNet3d',  # name of backbone
        pretrained2d=True,
        pretrained=
        'torchvision://resnet50',  # url/site of the pretrained model from ImageNet. NOTE: this is just an ImageNet & not Kinetics pre-trained model
        depth=50,
        conv_cfg=dict(type='Conv3d'),
        norm_eval=False,
        inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
        zero_init_residual=False),
    cls_head=dict(  # dict for classification head
        type='I3DHead',  # name of classification head
        num_classes=9,
        in_channels=2048,  # the input channels of classification head
        spatial_type='avg',  # type of pooling in spatial dimension
        dropout_ratio=0.5,  # probability in dropout layer
        init_std=0.01),  # std value for linear layer initiation
    # model training and testing settings
    train_cfg=None,  # config for training hyperparameters
    test_cfg=dict(average_clips='prob'))  # config for testing hyperparameters

### dataset settings
dataset_type = 'VideoDataset'  # Type of dataset for training, validation & testing. Can be video | frames | custom
# data_root = 'data/tanz/videos_train' # root path to data for training
# data_root_val = 'data/tanz/videos_val' # root path to data for validation & testing
data_root = ''
data_root_val = ''
ann_file_train = 'data/tanz/tanz_train_list_videos.txt'  # root path to annotations for training
ann_file_val = 'data/tanz/tanz_val_list_videos.txt'  # path to annotation files for validation
ann_file_test = '/mnt/data_transfer/read/to_process_test/tanz_test_list_videos.txt'  # path to annotation files for testing

img_norm_cfg = dict(  # config for image normalization used in data pipeline
    mean=[123.675, 116.28,
          103.53],  # mean values of different channels to normalize
    std=[58.395, 57.12,
         57.375],  # std values of different channels to normalize
    to_bgr=False)  # whether to convert channels from rgb to bgr

### The Pre-processing pipeline
### For the pre-processing pipeline down below, it might be a good idea to use lazy mode as this will accelerate the training
# checkout the documentation of every parameter: https://mmaction2.readthedocs.io/en/latest/api.html

train_pipeline = [  # list of training pipeline steps
    dict(type='DecordInit'),
    dict(  # config of sample frames
        type=
        'SampleFrames',  # sample frames pipeline, sampling frames from video
        clip_len=32,  # frames of each sampled output clip
        frame_interval=2,  # temporal interval of adjacent sampled frames
        num_clips=1),  # number of clips to be sampled
    dict(type='DecordDecode'),
    dict(
        type='Resize',  # resize pipeline
        scale=(-1, 256)),  # the scale to resize images
    dict(  # crop images with a list of randomly selected scales
        type='MultiScaleCrop',
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
    dict(type='FormatShape', input_format='NCTHW'
         ),  # format final image shape to the given input_format
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
        frame_interval=2,
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

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
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

data = dict(  # configs of data
    # Fix these two `_gpu` values according to the Linear Scaling Rule (https://arxiv.org/abs/1706.02677)
    # https://mmaction2.readthedocs.io/en/latest/recognition_models.html
    videos_per_gpu=8,  # number of videos in each GPU, i.e. batch size of GPU
    workers_per_gpu=
    4,  # workers (sub-processes) to pre-fetch data for each single gpu; (multithreaded loading for PyTorch)
    test_dataloader=dict(  # Additional config of test dataloader
        videos_per_gpu=
        1,  # has to be one-one else the mainz gpus won't hold and SIGKILL9 error results
        workers_per_gpu=1),
    train=dict(  # training dataset config
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

### optimizer settings
# optimizer
optimizer = dict(
    type='SGD', lr=0.005, momentum=0.9,
    weight_decay=0.0001)  # change from 0.01 to 0.005
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[20, 40])
total_epochs = 60
checkpoint_config = dict(interval=5)
evaluation = dict(  # Config of evaluation during training
    interval=5,  # Interval to perform evaluation
    metric_options=dict(top_k_accuracy=dict(
        topk=(1, 2, 3, 5))),  # set the top-k accuracy during validation;
    # for training the corresponding head must be modified (i3d_head in this case): https://github.com/open-mmlab/mmaction2/issues/874
    # for testing: eval config below
)
eval_config = dict(
    metric_options=dict(top_k_accuracy=dict(topk=(1, 2, 3, 5))), )

### runtime settings
# use the pre-trained model on kinetics400
load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth'
