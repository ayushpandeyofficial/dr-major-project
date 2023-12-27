import torchvision.transforms as T 


# Update the transforms to include CLAHE
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Normalize(mean=[0.489, 0.511, 0.511], std=[0.0783,0.0783,0.0783]),
])

