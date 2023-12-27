import torch
from torchvision import transforms as T
import gradio as gr
from src.models.resnet50 import ResNet50Model

labels = ["No_DR","Mild","Moderate","Proliferate_DR","Severe"]
model_path = r'artifacts/model_run-resnet50_2023-12-26-17-28-13/model_checkpoint.pt'
checkpoint = torch.load(model_path)

model_state_dict = checkpoint['model_state_dict']

model = ResNet50Model(num_labels=5)
model.load_state_dict(model_state_dict)
model.eval()

# Define the transformation for input images

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Normalize(mean=[0.489, 0.511, 0.511], std=[0.0783,0.0783,0.0783]),
])

def predict(inp):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(prediction.shape[0])}

    return confidences

gr.Interface(fn=predict,
            inputs=gr.Image(type="pil"),
            outputs=gr.Label(num_top_classes=5),
            examples=[r"data/Severe/0c917c372572.png",
                    "data/No_DR/0a4e1a29ffff.png"]).launch()
