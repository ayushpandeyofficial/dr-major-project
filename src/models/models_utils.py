from argparse import ArgumentParser

import torch

from src.models.densenet121 import DenseNet121Model
from src.models.efficientnetb1 import EfficientNetB1Model
from src.models.resnet50 import ResNet50Model
from src.models.vit_base_patch16_224 import VisionTransformerModel
from src.models.swin_transformer import (
    SwinTransformerModel,
)


def parse_arguments(training=True):
    parser = ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["resnet50", "efficientnet-b1", "swin_transformer", "densenet121","vit"],
        required=True,
    )

    if training:
        return parser.parse_args()

    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-f", "--folder-name", type=str, required=True)

    return parser.parse_args()


def load_model(model_name, num_labels, device):
    if model_name == "resnet50":
        model = ResNet50Model(num_labels=num_labels).to(device)
    elif model_name == "efficientnet-b1":
        model = EfficientNetB1Model(num_labels=num_labels).to(device)
    elif model_name == "swin_transformer":
        model = SwinTransformerModel(num_labels=num_labels).to(device)
    elif model_name == "densenet121":
        model = DenseNet121Model(num_labels=num_labels).to(device)
    elif model_name == "vit":
        model = VisionTransformerModel(num_labels=num_labels).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
