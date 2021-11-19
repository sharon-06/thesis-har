# Fix
The error comes from an incompatibility between the current environment & torch/CUDA/mmcv versions

In this case, the new GPUs don't work: `GPU 0,1: GeForce RTX 3090`

For the moment use the old GPUs: `GPU 2,3: GeForce RTX 2080 Ti`

# Mentioned in Issue
https://github.com/open-mmlab/mmdetection/issues/4335

# Error Log

```python
Performing Human Detection for each frame
  0%|                                                                                                                            | 0/876 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "demo/demo_spatiotemporal_det.py", line 410, in <module>
    main()
  File "demo/demo_spatiotemporal_det.py", line 314, in main
    human_detections = detection_inference(args, center_frames)
  File "demo/demo_spatiotemporal_det.py", line 220, in detection_inference
    result = inference_detector(model, frame_path)
  File "/opt/conda/lib/python3.8/site-packages/mmdet/apis/inference.py", line 147, in inference_detector
    results = model(return_loss=False, rescale=True, **data)
  File "/opt/conda/lib/python3.8/site-packages/torch/nn/modules/module.py", line 889, in _call_impl
    result = self.forward(*input, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 95, in new_func
    return old_func(*args, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/mmdet/models/detectors/base.py", line 169, in forward
    return self.forward_test(img, img_metas, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/mmdet/models/detectors/base.py", line 146, in forward_test
    return self.simple_test(imgs[0], img_metas[0], **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/mmdet/models/detectors/two_stage.py", line 178, in simple_test
    proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
  File "/opt/conda/lib/python3.8/site-packages/mmdet/models/dense_heads/rpn_test_mixin.py", line 36, in simple_test_rpn
    proposal_list = self.get_bboxes(*rpn_outs, img_metas)
  File "/opt/conda/lib/python3.8/site-packages/mmcv/runner/fp16_utils.py", line 182, in new_func
    return old_func(*args, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/mmdet/models/dense_heads/anchor_head.py", line 578, in get_bboxes
    result_list = self._get_bboxes(mlvl_cls_scores, mlvl_bbox_preds,
  File "/opt/conda/lib/python3.8/site-packages/mmdet/models/dense_heads/rpn_head.py", line 247, in _get_bboxes
    dets, keep = batched_nms(mlvl_proposals, mlvl_scores, mlvl_ids,
  File "/opt/conda/lib/python3.8/site-packages/mmcv/ops/nms.py", line 278, in batched_nms
    dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
  File "/opt/conda/lib/python3.8/site-packages/mmcv/utils/misc.py", line 310, in new_func
    output = old_func(*args, **kwargs)
  File "/opt/conda/lib/python3.8/site-packages/mmcv/ops/nms.py", line 144, in nms
    inds = NMSop.apply(boxes, scores, iou_threshold, offset)
  File "/opt/conda/lib/python3.8/site-packages/mmcv/ops/nms.py", line 19, in forward
    inds = ext_module.nms(
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
