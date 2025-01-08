#imports
import argparse
import os
import torch 
from torch import nn
import torch.optim as optim 
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader


#function to load dataset
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    #transforms 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229,0.224,0.225])
                                         ])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229,0.224, 0.225])
                                          ])
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_dataloader = DataLoader(validation_dataset, batch_size = 32, shuffle = False)
    
    return train_dataloader, val_dataloader, train_dataset.class_to_idx

#function to build the model
def build_model(architecture, output_size, hidden_unit1, hidden_unit2, hidden_unit3, hidden_unit4):
    # Selecting the architecture
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        input_size = 1280  # feature size for EfficientNet-B0
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        #the classifier
        model.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_unit1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit1, hidden_unit2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit2, hidden_unit3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit3, hidden_unit4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit4, output_size),
            nn.LogSoftmax(dim=1)
        )
    elif architecture == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        input_size = 512  # feature size for ResNet-18
        # Freeze parameters
        for param in model.parameters():
            param.requires_grad = False
        # Replace the fully connected layer (fc)
        model.fc = nn.Sequential(
            nn.Linear(input_size, hidden_unit1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit1, hidden_unit2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit2, hidden_unit3),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit3, hidden_unit4),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_unit4, output_size),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model


#function to train and evaluate the model 
def train_evaluate_model(model, train_loader, valid_loader, criterion, optimizer, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #backpropagation
            loss.backward() 
            optimizer.step()

            #update loss
            train_loss += loss.item()
         
        #validate
        model.eval()
        val_loss = 0
        accuracy = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                #update loss
                val_loss += loss.item()

                #calculating accuracy 
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(val_loss/len(valid_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(valid_loader)))

#function to save checkpoint
def save_checkpoint(model, optimizer, class_to_idx, save_dir, epochs):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "epochs": epochs,
    }
    torch.save(checkpoint, save_dir)
    print(f"Checkpoint saved to {save_dir}")
    
# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a new neural network: Image Classifier")
    parser.add_argument('data_dir', type=str, help="Directory of training data")
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help="Save directory for checkpoint")
    parser.add_argument('--architecture', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'resnet18'], help="Model architecture")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")

    #known arguments parsed first to determine architecture
    args, unknown_args = parser.parse_known_args()

    # Set default hidden units based on architecture
    if args.architecture == "efficientnet_b0":
        default_hidden_units = [1024, 512, 256, 128]
    elif args.architecture == "resnet18":
        default_hidden_units = [512, 256, 128, 64]

    # Add hidden unit arguments after determining architecture defaults
    parser.add_argument('--hidden_units1', type=int, default=default_hidden_units[0], help="First layer of Hidden units")
    parser.add_argument('--hidden_units2', type=int, default=default_hidden_units[1], help="Second layer of Hidden units")
    parser.add_argument('--hidden_units3', type=int, default=default_hidden_units[2], help="Third layer of Hidden units")
    parser.add_argument('--hidden_units4', type=int, default=default_hidden_units[3], help="Final layer of Hidden units")
    parser.add_argument('--output_size', type=int, default=102, help="Output size (number of classes)")

    # Reparse all arguments after adding hidden units
    args = parser.parse_args()

    # Set the device
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    # Load data
    train_loader, valid_loader, class_to_idx = load_data(args.data_dir)

    # Build model
    model = build_model(
        args.architecture, args.output_size,
        args.hidden_units1, args.hidden_units2, args.hidden_units3, args.hidden_units4
    )

    # Set optimizer and loss function
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()

    # Train and evaluate the model
    train_evaluate_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, device)

    # Save checkpoint
    save_checkpoint(model, optimizer, class_to_idx, args.save_dir, args.epochs)

if __name__ == '__main__':
    main()
