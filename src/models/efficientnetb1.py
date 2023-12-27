import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

class EfficientNetB1Model(nn.Module):
    def __init__(self, num_labels):
        super(EfficientNetB1Model, self).__init__()

        # Load pre-trained EfficientNet-B1 model
        self.efficientnet_b1 = EfficientNet.from_pretrained('efficientnet-b1')

        # Freeze all parameters in the pre-trained model
        for param in self.efficientnet_b1.parameters():
            param.requires_grad = False

        # Replace the final layer of the classifier
        in_features = self.efficientnet_b1._fc.in_features

        self.efficientnet_b1._fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.efficientnet_b1(x)

if __name__ == "__main__":
    # Create an instance of the model
    model = EfficientNetB1Model(num_labels=5)

    # Use torchsummary to print the model summary
    summary(model, input_size=(3, 224, 224))
