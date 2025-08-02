# Install required libraries (run this once)
# pip install transformers torch pillow gradio

import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import os
import numpy as np  
MODEL_PATH = "deepfake-detector-model-v1"  
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist. Please check the path.")
required_files = ["config.json", "model.safetensors", "preprocessor_config.json"]
if not all(os.path.exists(os.path.join(MODEL_PATH, f)) for f in required_files):
    raise FileNotFoundError(
        f"Model files not found in {MODEL_PATH}. Please ensure you have these files: "
        f"{', '.join(required_files)}"
    )
 
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = SiglipForImageClassification.from_pretrained(MODEL_PATH)
    print("Successfully loaded model from local directory")
except Exception as e:
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}. Error: {str(e)}")


id2label = {"0": "fake", "1": "real"}

def classify_image(image):
    """Classify an image as real or fake using the local deepfake detection model."""
    try:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        
        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        
        # Format the prediction
        prediction = {
            id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
        }
        
        return prediction
    except Exception as e:
        return {"error": str(e)}

# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Deepfake Classification"),
    title="Deepfake Image Detector (Local Model)",
    description="Upload an image to classify whether it is real or AI-generated using locally loaded model.",
    examples=[
        ["peakpx.jpg"],  # Add your own example files
  
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
