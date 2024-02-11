import torch.nn as nn
from torchvision import models

class DenseNet121Model(nn.Module):
    def __init__(self, num_labels):
        super(DenseNet121Model, self).__init__()

        # Load pre-trained DenseNet-121 model
        self.densenet121 = models.densenet121(pretrained=True)

        # Freeze all parameters in the pre-trained model
        for param in self.densenet121.parameters():
            param.requires_grad = False

        # Define additional layers
        self.densenet121.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(2048),
            nn.Dropout(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.densenet121(x)

if __name__ == "__main__":
    # Create an instance of the model
    model = DenseNet121Model(num_labels=8)

    # Print the model structure
    print(model)

    # Count the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
