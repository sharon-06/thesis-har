# Problem
Sometimes the training process goes very slow

Diagnostification steps:
- check cpu usage: htop, to see if cpus are busy
- check disk usage: nmon and press d, to see if ssd/hdd are busy

if cpu is busy, then the bottleneck is pre-processing
if disk is busy, then the bottleneck is io

ideally, these two are in low-usage, and gpus are running at 100%

# Fix

In this case the bottleneck was with cpu, which means that it is with the pre-processing steps.

Checkout HighCpuConsumption for the solution.

# Related Issues
1. https://github.com/open-mmlab/mmaction2/issues/264
2. https://github.com/open-mmlab/mmaction2/issues/949
3. https://github.com/open-mmlab/mmaction2/issues/923
4. https://github.com/open-mmlab/mmaction2/issues/997
5. https://github.com/open-mmlab/mmaction2/issues/421
6. https://github.com/pytorch/pytorch/issues/1838
