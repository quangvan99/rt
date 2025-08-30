trtexec --onnx=slow_only.onnx --saveEngine=slow_only_dy.engine --fp16
trtexec --onnx=yolov8n_backbone.onnx --saveEngine=yolov8n_backbone.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16

trtexec --onnx=slow_only.onnx --saveEngine=slow_only_dy.engine --fp16 --minShapes=proposal:4x4 --optShapes=proposal:8x4 --maxShapes=proposal:16x4 