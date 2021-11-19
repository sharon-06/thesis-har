# Fix
When using `cmake` to build `OpenCv` with Cuda suport you must pay attention to set the `CUDA_GENERATION` to the specific GPU architecture.

For e.g., for the older Geforce RTX 2080 Ti GPU you need to set `CUDA_GENERATION="Turing"`

# Mentioned in issue
[Link](https://github.com/open-mmlab/denseflow/issues/39)

# Error Log & Description

Error happens when using the [Dockerfile](https://github.com/open-mmlab/denseflow/blob/master/docker/Dockerfile) for Denseflow.
(Setting up Denseflow with a Dockerfile can be better than the usual setup)

Full Error:

```python
terminate called after throwing an instance of 'cv::Exception'
  what():  OpenCV(4.5.2) /opencv_contrib/modules/cudev/include/opencv2/cudev/grid/detail/transform.hpp:312: error: (-217:Gpu API call) no kernel image is available for execution on the device in function 'call'

Aborted (core dumped)
```
