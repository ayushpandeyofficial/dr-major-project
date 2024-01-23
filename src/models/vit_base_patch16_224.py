import torch.nn as nn
from timm import create_model
from torchsummary import summary

class VisionTransformerModel(nn.Module):
    def __init__(self, num_labels):
        super(VisionTransformerModel, self).__init__()

        # Load pre-trained Vision Transformer model
        self.vit_model = create_model('vit_base_patch16_224', pretrained=True)

        # Freeze all parameters in the pre-trained model
        for param in self.vit_model.parameters():
            param.requires_grad = False

        # Modify the classification head
        in_features = self.vit_model.head.in_features
        
        self.vit_model.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.vit_model(x)

if __name__ == "__main__":
    # Create an instance of the model
    model = VisionTransformerModel(num_labels=5)

    # Use torchsummary to print the model summary
    summary(model, input_size=(3, 224, 224))
