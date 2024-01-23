import torch
import torch.nn as nn
from torchsummary import summary
from timm import create_model

class SwinTransformerModel(nn.Module):
    def __init__(self, num_labels, freeze_backbone=True):
        super(SwinTransformerModel, self).__init__()

        # Load pre-trained Swin Transformer model
        self.swin_transformer = create_model('swin_base_patch4_window7_224', pretrained=True)

        # Freeze or unfreeze the parameters based on the flag
        for param in self.swin_transformer.parameters():
            param.requires_grad = not freeze_backbone

        # Modify the classifier head
        in_features = self.swin_transformer.head.in_features

        self.swin_transformer.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        return self.swin_transformer(x)


if __name__ == "__main__":
    # Create an instance of the Swin Transformer model
    model = SwinTransformerModel(num_labels=5)

    # Use torchsummary to print the model summary
    summary(model, input_size=(3, 224, 224))
