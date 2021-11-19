# Fix

This seems to have been a memory related problem. GPU-memory exhaustion.

We have 4 GPUs available. 2x Geforce RTX 3090 `24gb` each & 2x Geforce RTX 2080 Ti `11gb` each. The latter are connected through p2p memory pooling but the former are not, so more than 30gb cannot be harnessed.

Fine-tune `videos_per_gpu` & `workers_per_gpu` in the config file in order to lower the load to the GPUs.


# Error
`Signal.SIGKILL9`

Training doesn't start

```python
traceback (most recent call last):
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
subprocess.CalledProcessError: Command '['/opt/conda/bin/python', '-u', 'tools/train.py', '--local_rank=3', 'configs/recognition/i3d/i3d_r50_video_32x2x1_tanz_v1.py', '--launcher', 'pytorch', '--work-dir', '../mnt/data_transfer/write/work_dir3/', '--validate']' died with <Signals.SIGKILL: 9>.
```
