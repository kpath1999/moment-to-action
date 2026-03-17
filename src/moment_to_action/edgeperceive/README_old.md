# Edge Inference Framework 

## Implemented as a sequence of stages with message passing between them

Run command:
'''
cd ~/moment_to_action
uv run python -m moment_to_action.edgeperceive.pipeline.draw_detections   --image ../../../../edgeperception/fighting.jpg --model ../models/yolo/model.onnx
'''

Using a confidence score for YoLo:
'''
uv run python -m moment_to_action.edgeperceive.pipeline.run_yolo_pipeline   --image ../../../../edgeperception/fighting.jpg --model ../models/yolo/model.onnx --conf 0.1
'''

Here's how I would suggest going through the files:


