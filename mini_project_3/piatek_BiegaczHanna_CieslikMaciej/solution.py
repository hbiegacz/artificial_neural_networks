import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, field
from typing import List, Tuple
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'train'))
TESTING_DATA_PATH  = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'test'))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'pred.csv')

IMG_SIZE = 64

@dataclass
class NetConfig:
    convolutional_layers: List[int]   = field(default_factory=lambda: [32, 64, 128, 256])
    fully_connected_layers: List[int] = field(default_factory=lambda: [512, 128])
    dropout_rates: List[float]        = field(default_factory=lambda: [0.5])
    activation_types: List[type]      = field(default_factory=lambda: [nn.ReLU])
    use_batch_normalization: bool     = True
    batch_size: int                   = 64
    epochs: int                       = 25
    learning_rate: float              = 1e-3
    optimizer_type: str               = "Adam" 
    weight_decay: float               = 1e-5
    momentum: float                   = 0.9
    nesterov: bool                    = False

    def expand_parameter_for_layers(self, parameter_list: list, layer_count: int) -> list:
        if len(parameter_list) == 1:
            return parameter_list * layer_count
        return parameter_list


class UnlabeledImageDataset(Dataset):
    def __init__(self, root_directory, transform=None):
        self.root_directory = root_directory
        self.transform = transform
        self.image_filenames = sorted([
            filename for filename in os.listdir(root_directory) 
            if filename.lower().endswith(('.jpeg', '.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        filename = self.image_filenames[index]
        image_path = os.path.join(self.root_directory, filename)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, filename



class NeuralNetwork:
    def __init__(self, config: NetConfig, num_classes: int = 50):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.initialize_network(num_classes).to(self.device)

    def initialize_network(self, num_classes: int) -> nn.Sequential:
        network_layers = []
        current_channels = 3

        for output_channels in self.config.convolutional_layers:
            network_layers.append(nn.Conv2d(current_channels, output_channels, kernel_size=3, padding=1))
            if self.config.use_batch_normalization:
                network_layers.append(nn.BatchNorm2d(output_channels))
            network_layers.append(nn.ReLU())
            network_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = output_channels

        network_layers.append(nn.Flatten())

        image_dimension = IMG_SIZE // (2 ** len(self.config.convolutional_layers))
        flattened_size = current_channels * (image_dimension ** 2)

        num_fc = len(self.config.fully_connected_layers)
        dropouts = self.config.expand_parameter_for_layers(self.config.dropout_rates, num_fc)
        activations = self.config.expand_parameter_for_layers(self.config.activation_types, num_fc)

        current_dimension = flattened_size
        for hidden_units, dropout_rate, activation_type in zip(self.config.fully_connected_layers, dropouts, activations):
            network_layers.append(nn.Linear(current_dimension, hidden_units))
            if self.config.use_batch_normalization:
                network_layers.append(nn.BatchNorm1d(hidden_units))
            network_layers.append(activation_type())
            if dropout_rate > 0:
                network_layers.append(nn.Dropout(dropout_rate))
            current_dimension = hidden_units

        network_layers.append(nn.Linear(current_dimension, num_classes))
        return nn.Sequential(*network_layers)



    def fit(self, training_loader: DataLoader, print_epochs: bool =False):
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        opt_type = self.config.optimizer_type

        if opt_type == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "AdamW":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_type == "SGD":
            momentum = getattr(self.config, 'momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
            
        loss_function = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        for epoch in range(self.config.epochs):
            self.model.train()
            running_loss, correct_predictions, total_samples = 0.0, 0, 0

            for images, labels in training_loader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                    outputs = self.model(images)
                    loss = loss_function(outputs, labels)

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * images.size(0)
                correct_predictions += outputs.argmax(dim=1).eq(labels).sum().item()
                total_samples += labels.size(0)

            average_loss = running_loss / total_samples
            accuracy = 100.0 * correct_predictions / total_samples
            if print_epochs: print(f"\tEpoch {epoch+1}/{self.config.epochs} - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")


    def predict(self, data_loader: DataLoader) -> List[Tuple[str, int]]:
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for images, meta in data_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                predicted_classes = outputs.argmax(dim=1).cpu().numpy()

                if isinstance(meta, torch.Tensor):
                    meta = meta.cpu().numpy()

                for m, p in zip(meta, predicted_classes):
                    all_predictions.append((m, int(p)))

        return all_predictions



def read_trainset(path: str = TRAINING_DATA_PATH, augmentation: str = 'none', img_size: int = IMG_SIZE):
    common_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # DATA AUGMENTATION
    aug_transforms = []
    if augmentation == 'weak':
        aug_transforms = [transforms.RandomHorizontalFlip(p=0.5)]
    elif augmentation == 'strong':
        aug_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0))
        ]

    training_transforms = transforms.Compose(aug_transforms + common_transforms)

    dataset = torchvision.datasets.ImageFolder(root=path, transform=training_transforms)
    return dataset.class_to_idx, dataset



def read_testset(path: str = TESTING_DATA_PATH, img_size: int = IMG_SIZE):
    testing_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return UnlabeledImageDataset(root_directory=path, transform=testing_transforms)

def save_predictions_to_csv(predictions: List[Tuple[str, int]], output_path: str = OUTPUT_PATH):
    with open(output_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(predictions)



def main():
    final_config = NetConfig(epochs=10, use_batch_normalization=True)
    if torch.cuda.is_available(): print(f"\n--- Using GPU ({torch.cuda.get_device_name(0)}) ---\n")
    else: print("\n--- Using CPU ---\n")

    print("[Step 1/4] Loading datasets...")
    _, train_dataset = read_trainset(augmentation="none", img_size=IMG_SIZE)
    test_dataset = read_testset()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=final_config.batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=final_config.batch_size, 
        shuffle=False,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,          
        persistent_workers=True     
    )

    print("\n[Step 2/4] Initializing and training the model...")
    network = NeuralNetwork(final_config, num_classes=len(train_dataset.classes))
    network.fit(train_loader, print_epochs=True)

    print("\n[Step 3/4] Generating predictions for test set...")
    predictions = network.predict(test_loader)

    print("\n[Step 4/4] Saving results to CSV...")
    save_predictions_to_csv(predictions)
    print(f"--- All tasks completed! Final predictions saved to {OUTPUT_PATH} ---\n")

if __name__ == "__main__":
    main()
