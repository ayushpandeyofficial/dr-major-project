# import torch.nn as nn
# from timm import create_model
# from torchsummary import summary

# class VisionTransformerModel(nn.Module):
#     def __init__(self, num_labels):
#         super(VisionTransformerModel, self).__init__()

#         # Load pre-trained Vision Transformer model
#         self.vit_model = create_model('vit_base_patch16_224', pretrained=True)

#         # Freeze all parameters in the pre-trained model
#         for param in self.vit_model.parameters():
#             param.requires_grad = False

#         # Modify the classification head
#         in_features = self.vit_model.head.in_features
        
#         self.vit_model.head = nn.Sequential(
#             nn.Linear(in_features, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, num_labels)
#         )

#     def forward(self, x):
#         return self.vit_model(x)

# if __name__ == "__main__":
#     # Create an instance of the model
#     model = VisionTransformerModel(num_labels=5)

#     # Use torchsummary to print the model summary
#     summary(model, input_size=(3, 224, 224))


import torch
import torch.nn as nn
import torchvision
from torchinfo import summary

class CustomViT(nn.Module):
    def __init__(self, num_labels, pretrained_weights=torchvision.models.ViT_B_16_Weights.DEFAULT):
        super(CustomViT, self).__init__()
        self.pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_weights)
        self.freeze_base_parameters()
        self.modify_classifier_head(num_labels)

    def freeze_base_parameters(self):
        for parameter in self.pretrained_vit.parameters():
            parameter.requires_grad = False

    def modify_classifier_head(self, num_labels):
        self.pretrained_vit.heads = nn.Linear(in_features=768, out_features=num_labels)

    def forward(self, x):
        return self.pretrained_vit(x)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["No_DR","Mild","Moderate","Severe","Proliferate_DR"]
    custom_vit_model = CustomViT(num_labels=len(class_names)).to(device)
    summary(model=custom_vit_model, 
            input_size=(32, 3, 224, 224), 
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )