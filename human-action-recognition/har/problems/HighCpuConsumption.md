# Problem
Ideally, the training process should mostly use the GPUs. If the CPU is utilized to its fullest, there is a problem.

# Try

1. @`train.py`:
-  `import cv2; cv2.setNumThreads(1)` or `setNumThreads(0)`

2. Reduce the batch size (vdeos per gpu)

3. Reduce the num_workers

4. `resize_videos.py` might also help since CPU is the bottleneck: https://mmaction2.readthedocs.io/en/latest/prepare_data.html?highlight=resize_videos.py#id58

5.  While testing the model, less CPU power is being used. Hence examine the differences between test_ & train_pipeline. E.g. `num_clips`

# Fix

In this case, it was fixed using `workers_per_gpu=0` in the config file

# Related Issues
1. https://github.com/open-mmlab/mmaction2/issues/264
2. https://github.com/open-mmlab/mmaction2/issues/949
3. https://github.com/open-mmlab/mmaction2/issues/923
4. https://github.com/open-mmlab/mmaction2/issues/997
5. https://github.com/open-mmlab/mmaction2/issues/421
6. https://github.com/pytorch/pytorch/issues/1838
