
# Fix
https://github.com/open-mmlab/mmaction2/issues/855

# Error Log:

`ImportError: cannot import name 'MODELS' from 'mmcv.cnn' (/opt/conda/lib/python3.8/site-packages/mmcv/cnn/__init__.py)`

```python
/mmaction2/mmaction/core/evaluation/eval_hooks.py:36: UserWarning: DeprecationWarning: EvalHook and DistEvalHook in mmaction2 will be deprecated, please install mmcv through master branch.
  warnings.warn('DeprecationWarning: EvalHook and DistEvalHook in mmaction2 '
Traceback (most recent call last):
  File "demo/demo_gradcam.py", line 10, in <module>
    from mmaction.apis import init_recognizer
  File "/mmaction2/mmaction/apis/__init__.py", line 1, in <module>
    from .inference import inference_recognizer, init_recognizer
  File "/mmaction2/mmaction/apis/inference.py", line 12, in <module>
    from mmaction.models import build_recognizer
  File "/mmaction2/mmaction/models/__init__.py", line 1, in <module>
    from .backbones import (C3D, X3D, MobileNetV2, MobileNetV2TSM, ResNet,
  File "/mmaction2/mmaction/models/backbones/__init__.py", line 1, in <module>
    from .c3d import C3D
  File "/mmaction2/mmaction/models/backbones/c3d.py", line 7, in <module>
    from ..builder import BACKBONES
  File "/mmaction2/mmaction/models/builder.py", line 3, in <module>
    from mmcv.cnn import MODELS as MMCV_MODELS
ImportError: cannot import name 'MODELS' from 'mmcv.cnn' (/opt/conda/lib/python3.8/site-packages/mmcv/cnn/__init__.py)
```
