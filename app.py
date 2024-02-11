# import torch
# from torchvision import transforms as T
# import gradio as gr
# from src.models.resnet50 import ResNet50Model

# labels = ["No_DR","Mild","Moderate","Proliferate_DR","Severe"]
# model_path = r'artifacts/model_run-resnet50_2023-12-26-17-28-13/model_checkpoint.pt'
# checkpoint = torch.load(model_path)

# model_state_dict = checkpoint['model_state_dict']

# model = ResNet50Model(num_labels=5)
# model.load_state_dict(model_state_dict)
# model.eval()

# # Define the transformation for input images

# transform = T.Compose([
#     T.Resize((224, 224)),
#     T.ToTensor(),
#     # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     T.Normalize(mean=[0.489, 0.511, 0.511], std=[0.0783,0.0783,0.0783]),
# ])

# def predict(inp):
#     inp = transform(inp).unsqueeze(0)
#     with torch.no_grad():
#         prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
#         confidences = {labels[i]: float(prediction[i]) for i in range(prediction.shape[0])}

#     return confidences

# gr.Interface(fn=predict,
#             inputs=gr.Image(type="pil"),
#             outputs=gr.Label(num_top_classes=5),
#             examples=[r"data/Severe/0c917c372572.png",
#                     "data/No_DR/0a4e1a29ffff.png"]).launch()

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import gradio as gr
import cv2
from src.models.resnet50 import ResNet50Model

# Load the model
model = ResNet50Model(num_labels=5)
model_path = 'artifacts/model_run-resnet50_2023-12-26-17-28-13/model_checkpoint.pt'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Labels for the classification
labels = ["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"]

# Define the transformation for input images
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.489, 0.511, 0.511], std=[0.0783, 0.0783, 0.0783]),
])

# Function to generate Grad-CAM heatmap
def generate_gradcam(model, inp, target_class):
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.cpu().data.numpy())
        
    # Get the output of the last convolutional layer
    last_conv_output = None
    def hook_last_conv(module, input, output):
        nonlocal last_conv_output
        last_conv_output = output.detach()
    last_conv_hook = model.resnet50.layer4.register_forward_hook(hook_last_conv)
    
    # Forward pass
    model(inp)
    
    # Remove hook
    last_conv_hook.remove()
    
    # Get gradients for the target class
    model.zero_grad()
    prediction = model(inp)
    prediction[:, target_class].backward()
    
    # Compute the heatmap
    grads = last_conv_output.grad.mean(dim=(2, 3), keepdim=True)
    heatmap = torch.mean(grads * last_conv_output, dim=1, keepdim=True)
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.squeeze().cpu().numpy()
    
    return heatmap

def predict_with_heatmap(inp):
    # Convert the input to PIL image if it's not already
    if not isinstance(inp, Image.Image):
        inp = Image.fromarray(inp)
    
    inp_tensor = transform(inp).unsqueeze(0)
    
    # Generate prediction
    prediction = predict(inp)
    pred_class = np.argmax(prediction)
    
    # Generate Grad-CAM heatmap for the predicted class
    heatmap = generate_gradcam(model, inp_tensor, pred_class)
    
    # Rescale heatmap to the size of the input image
    heatmap = cv2.resize(heatmap, (inp.size[0], inp.size[1]))
    
    # Apply heatmap on the input image
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(cv2.cvtColor(np.array(inp), cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)
    
    # Display the images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(inp)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Heatmap')
    plt.axis('off')
    
    plt.show()

# Define the function for prediction
def predict(inp):
    inp = transform(inp).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in range(prediction.shape[0])}

    return confidences

# Interface for Gradio
gr.Interface(fn=predict_with_heatmap,
            inputs="image",
            outputs=None,
            examples=[r"data/Severe/0c917c372572.png",
                    "data/No_DR/0a4e1a29ffff.png"]).launch()

