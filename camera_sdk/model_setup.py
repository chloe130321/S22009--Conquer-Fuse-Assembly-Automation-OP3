import torch
from timm import create_model

def get_model(num_classes=2):
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pretrained ConvNeXt model
    model = create_model("convnext_tiny", pretrained=True, num_classes=num_classes)
    model = model.to(device)

    return model
