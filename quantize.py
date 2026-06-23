from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="/app/model/onnx/model.onnx",
    model_output="/app/model/onnx/model.int8.onnx",
    weight_type=QuantType.QInt8
)
