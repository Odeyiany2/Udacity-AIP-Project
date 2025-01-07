#imports 
import argparse
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json
from torch import nn

#function to load model checkpoint 
def load_checkpoint(filepath, model_name):
    checkpoint = torch.load(filepath)
    
    # Dynamically choose the model architecture
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        input_units = 1280  # EfficientNet-B0 final feature size
    elif model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        input_units = 512   # ResNet-18 final feature size
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    # Replace classifier or fully connected layer
    if model_name == "efficientnet_b0":
        model.classifier = nn.Sequential(
            nn.Linear(input_units, checkpoint["hidden_units1"]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint["hidden_units1"], checkpoint["hidden_units2"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(checkpoint["hidden_units2"], checkpoint["output_size"]),
            nn.LogSoftmax(dim=1)
        )
    elif model_name == "resnet18":
        model.fc = nn.Sequential(
            nn.Linear(input_units, checkpoint["hidden_units1"]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(checkpoint["hidden_units1"], checkpoint["hidden_units2"]),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(checkpoint["hidden_units2"], checkpoint["output_size"]),
            nn.LogSoftmax(dim=1)
        )

    # Load state dict and class mappings
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

#function to preprocess an image
def process_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    
    #resize the shortest side to 256 while still keeping the aspect ratio
    width, height = pil_image.size
    if width < height:
        new_width, new_height = 256, int(256 * height/width)
    else:
        new_width, new_height = int(256 * width/height), 256
    
    pil_image = pil_image.resize((new_width, new_height))
    
    #cropping the center to 224 by 224
    left = (new_width - 224)/2
    upper = (new_height - 224)/2
    right = left + 224
    lower = upper + 224
    
    pil_image = pil_image.crop((left, upper, right, lower))
    
    #convert to numpy array and normalize
    np_image = np.array(pil_image)/255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    #reorder the dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.tensor(np_image).float()

#function to predict the class of an image
def predict(image_path, model, topk, device):
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    #evaluation
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output) #probabilities
        
        #getting the topk probabilities
        top_p, top_indices = ps.topk(topk, dim = 1)

    #converting the indices to classes
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices[0].tolist()]
    
    return top_p[0].tolist(), top_classes

# Function to load category names from a JSON file
def load_category_names(filepath):
    try:
        with open(filepath, 'r') as f:
            category_names = json.load(f)
        return category_names
    except Exception as e:
        print(f"Error loading category names: {e}")
        return None
    
# Main function
def main():
    parser = argparse.ArgumentParser(description="Predict image class using a trained model")
    parser.add_argument('image_path', type=str, help="Path to input image")
    parser.add_argument('checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--model', type=str, default="efficientnet_b0", choices=["efficientnet_b0", "resnet18"],
                        help="Model architecture to use (default: efficientnet_b0)")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K predictions")
    parser.add_argument('--category_names', type=str, default=None, help="Path to JSON file for category names")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")

    args = parser.parse_args()
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    # Validate top_k
    if args.top_k < 1:
        print("Error: top_k must be greater than or equal to 1")
        return

    # Load model
    model = load_checkpoint(args.checkpoint, args.model)

    # Load category names
    category_names = {}
    if args.category_names:
        category_names = load_category_names(args.category_names)

    # Make prediction
    top_probs, top_classes = predict(args.image_path, model, args.top_k, device)

    # Map class labels to names if category names are loaded
    if category_names:
        top_classes = [category_names.get(cls, cls) for cls in top_classes]

    print("Top Probabilities:", top_probs)
    print("Top Classes:", top_classes)