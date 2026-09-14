# Browser default is `web/model.onnx`

Fast currently loads `web/policy.onnx` while training exports `model.onnx`, so Human matches can sit on a stale file. After a run whose Loop eval has moved, copy `checkpoints/small/model.onnx` to `web/model.onnx` and make that the page default. Kaggle packaging is out of this pass.
