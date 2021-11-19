
# Fix
Downgrade decord from 0.5.0 to 0.4.0

Or alternatively one must find the error-videos and fix them. Finding these error videos can be quite challenging. Take a look at the `detect_corrupted_videos.py` script.

Moreover, according to [this](https://github.com/open-mmlab/mmaction2/issues/564) issue, using `ffmpeg_extract_sublip` is not the best idea

# Metioned in Issue
https://github.com/open-mmlab/mmaction2/issues/362

# Error Log

```python
INFO:mmaction:Epoch [5][20/84]  lr: 5.000e-03, eta: 5:51:34, time: 5.600, data_time: 3.118, memory: 5157, top1_acc: 0.6328, top5_acc: 0.9234, loss_cls: 1.0494, loss: 1.0494, grad_norm: 6.0283
Killing subprocess 186371
Killing subprocess 186372
Killing subprocess 186373
Killing subprocess 186374
Traceback (most recent call last):
  File "/opt/conda/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/opt/conda/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "/opt/conda/lib/python3.8/site-packages/torch/distributed/launch.py", line 340, in <module>
    main()
  File "/opt/conda/lib/python3.8/site-packages/torch/distributed/launch.py", line 326, in main
    sigkill_handler(signal.SIGTERM, None)  # not coming back
  File "/opt/conda/lib/python3.8/site-packages/torch/distributed/launch.py", line 301, in sigkill_handler
    raise subprocess.CalledProcessError(returncode=last_return_code, cmd=cmd)
subprocess.CalledProcessError: Command '['/opt/conda/bin/python', '-u', 'tools/train.py', '--local_rank=3', 'configs/recognition/i3d/i3d_r50_video_32x2x1_tanz_v1_0.7.py', '--launcher', 'pytorch', '--work-dir', '../mnt/data_transfer/write/work_dir2/', '--validate']' died with <Signals.SIGKILL: 9>.
root@0d47d9ec4654:/mmaction2# /opt/conda/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 20 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/opt/conda/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 20 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/opt/conda/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 20 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
/opt/conda/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 20 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '

root@0d47d9ec4654:/mmaction2#
```
