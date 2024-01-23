import torchvision.transforms as T 

def crop_nonzero(image):
    bbox = image.getbbox()
    cropped_image = image.crop(bbox)
    return cropped_image
    
class CropNonzero(object):
    def __call__(self, image):
        return crop_nonzero(image)

train_transform = T.Compose([
    
    # T.RandomAffine(degrees=30, shear=30), 
    # T.RandomHorizontalFlip(p=0.5),
    CropNonzero(),
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # T.Normalize(mean=[0.489, 0.511, 0.511], std=[0.0783,0.0783,0.0783]),
    T.Normalize(mean=[0.101, 0.104,0.106], std=[0.214,0.214,0.214]),

    
])
val_transform = T.Compose([
    CropNonzero(),
    T.Resize((224, 224)),
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # T.Normalize(mean=[0.489, 0.511, 0.511], std=[0.0783,0.0783,0.0783]),
    T.Normalize(mean=[0.101, 0.104,0.106], std=[0.214,0.214,0.214]),
])

