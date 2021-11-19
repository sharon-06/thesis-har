# Naming Conventions

{xxx} is required field and [yyy] is optional

e.g. (1) `tsm_r50_video_1x1x8_100e_bast.py`; (2) `csn_ig65m_r152_32x2x1_100e_bast_rgb_flow.py`

- {model}: model type, e.g. tsm, i3d, etc.
- [model setting]: specific setting for some models.
- {backbone}: backbone type, e.g. r50 (ResNet-50), etc.
- [misc]: miscellaneous setting/plugins of model, e.g. dense, 320p, video, etc.
- {data setting}: frame sample setting in {clip_len}x{frame_interval}x{num_clips} format.
- [gpu x batch_per_gpu]: GPUs and samples per GPU.
- {schedule}: training schedule, e.g. 20e means 20 epochs.
- {dataset}: dataset name, e.g. kinetics400, mmit, etc.
- {modality}: frame modality, e.g. rgb, flow, etc.
