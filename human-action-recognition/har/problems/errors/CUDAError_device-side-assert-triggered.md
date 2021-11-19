# Fix
The number of classes should be in the range [0, n_classes) as defined in the config file

# Mentioned in Issue

https://github.com/pytorch/pytorch/issues/1204

# Error Log:
`RuntimeError: CUDA error: device-side assert triggered terminate called after throwing an instance of 'c10::Error'`

`/opt/conda/conda-bld/pytorch_1614378083779/work/aten/src/THCUNN/ClassNLLCriterion.cu:108: cunn_ClassNLLCriterion_updateOutput_kernel: block: [0,0,0], thread: [6,0,0] Assertion t >= 0 && t < n_classes failed.`

```python
RuntimeError: CUDA error: device-side assert triggered
terminate called after throwing an instance of 'c10::Error'
  what():  CUDA error: device-side assert triggered
Exception raised from create_event_internal at /opt/conda/conda-bld/pytorch_1614378083779/work/c10/cuda/CUDACachingAllocator.cpp:733 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x42 (0x7f26794612f2 in /opt/conda/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x5b (0x7f267945e67b in /opt/conda/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #2: c10::cuda::CUDACachingAllocator::raw_delete(void*) + 0x809 (0x7f26796ba219 in /opt/conda/lib/python3.8/site-packages/torch/lib/libc10_cuda.so)
frame #3: c10::TensorImpl::release_resources() + 0x54 (0x7f26794493a4 in /opt/conda/lib/python3.8/site-packages/torch/lib/libc10.so)
frame #4: <unknown function> + 0x6e0dda (0x7f26d03bddda in /opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #5: <unknown function> + 0x6e0e71 (0x7f26d03bde71 in /opt/conda/lib/python3.8/site-packages/torch/lib/libtorch_python.so)
frame #6: <unknown function> + 0x1cab93 (0x55b6a3f42b93 in /opt/conda/bin/python)
frame #7: <unknown function> + 0x12b4d6 (0x55b6a3ea34d6 in /opt/conda/bin/python)
frame #8: <unknown function> + 0x12b8c6 (0x55b6a3ea38c6 in /opt/conda/bin/python)
frame #9: <unknown function> + 0x12b8c6 (0x55b6a3ea38c6 in /opt/conda/bin/python)
frame #10: <unknown function> + 0x11f850 (0x55b6a3e97850 in /opt/conda/bin/python)
frame #11: <unknown function> + 0x12bc96 (0x55b6a3ea3c96 in /opt/conda/bin/python)
frame #12: <unknown function> + 0x12bc4c (0x55b6a3ea3c4c in /opt/conda/bin/python)
frame #13: <unknown function> + 0x12bc4c (0x55b6a3ea3c4c in /opt/conda/bin/python)
frame #14: <unknown function> + 0x12bc4c (0x55b6a3ea3c4c in /opt/conda/bin/python)
frame #15: <unknown function> + 0x12bc4c (0x55b6a3ea3c4c in /opt/conda/bin/python)
frame #16: <unknown function> + 0x12bc4c (0x55b6a3ea3c4c in /opt/conda/bin/python)
frame #17: <unknown function> + 0x12bc4c (0x55b6a3ea3c4c in /opt/conda/bin/python)
frame #18: <unknown function> + 0x154ec8 (0x55b6a3eccec8 in /opt/conda/bin/python)
frame #19: PyDict_SetItemString + 0x87 (0x55b6a3ece127 in /opt/conda/bin/python)
frame #20: PyImport_Cleanup + 0x9a (0x55b6a3fce5aa in /opt/conda/bin/python)
frame #21: Py_FinalizeEx + 0x7d (0x55b6a3fce94d in /opt/conda/bin/python)
frame #22: Py_RunMain + 0x110 (0x55b6a3fcf7f0 in /opt/conda/bin/python)
frame #23: Py_BytesMain + 0x39 (0x55b6a3fcf979 in /opt/conda/bin/python)
frame #24: __libc_start_main + 0xe7 (0x7f270715cbf7 in /lib/x86_64-linux-gnu/libc.so.6)
frame #25: <unknown function> + 0x1e7185 (0x55b6a3f5f185 in /opt/conda/bin/python)
```
