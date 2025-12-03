import os
import torch
from PIL import Image
from torchvision import transforms

def predict_image(image_path, model, class_names, transform):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()  # Set model to evaluation mode

    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)

        return class_names[preds]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def predict_folder(folder_path, model, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the same transformations used during validation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    predictions = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            pred = predict_image(file_path, model, class_names, transform)
            predictions[filename] = pred

    return predictions
