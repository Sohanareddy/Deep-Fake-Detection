# Install required libraries (run this in your terminal or notebook)
# !pip install -q transformers torch pillow gradio hf_xet

import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import numpy as np

# Load model and processor
model_name = "prithivMLmods/deepfake-detector-model-v1"  
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "fake",
    "1": "real"
}

def classify_image(image):
    """
    Classify an image as real or fake using the deepfake detection model.
    
    Args:
        image: numpy array of the input image
        
    Returns:
        Dictionary with probabilities for 'fake' and 'real' classes
    """
    # Convert numpy array to PIL Image
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

# Create Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2, label="Deepfake Classification"),
    title="Deepfake Image Detector",
    description="Upload an image to classify whether it is real or AI-generated (deepfake).",
    examples=[
        ["peakpx.jpg"]  # You can add example image paths here
    ],
    allow_flagging="never"
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()
