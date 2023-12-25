import torch.nn as nn
from torchvision import models
from torchsummary import summary

class ResNet50Model(nn.Module):
    def __init__(self, num_labels):
        super(ResNet50Model, self).__init__()

        # Load pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)

        # Freeze all parameters in the pre-trained model
        for param in self.resnet50.parameters():
            param.requires_grad = False

        # Replace the final layer of the classifier
        in_features = self.resnet50.fc.in_features
        
        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.resnet50(x)

if __name__ == "__main__":
    # Create an instance of the model
    model = ResNet50Model(num_labels=5)

    # Use torchsummary to print the model summary
    summary(model, input_size=(3, 224, 224))