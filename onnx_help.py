import onnx
from onnx import helper

# Load ONNX model
model_path = "weights/ruun.onnx"
onnx_model = onnx.load(model_path)

# Find and modify ScatterND nodes
for node in onnx_model.graph.node:
    if node.op_type == "ScatterND":
        print(f"Modifying ScatterND node: {node.name}")

        # Add a new attribute for reduction mode
        reduction_attr = helper.make_attribute("reduction", "add")  # Change "add" to "max" or "min" if needed
        node.attribute.append(reduction_attr)

# Save modified model
onnx.save(onnx_model, "ruun_1.onnx")
print("Updated ONNX model saved as 'modified_model.onnx'")
