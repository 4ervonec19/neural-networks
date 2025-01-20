import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

import torch.nn as nn 
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tqdm.auto import tqdm
import os

# Define transforms used in research

# Classic one
base_transform_64 = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Which was applied to extract left upper corner of image as background
left_upper_corner_transfrom = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 0, 200, 200))),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
])

# Base augentation used for classic transforms
base_augmentation = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 0, 256, 256))),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), 
            transforms.RandomHorizontalFlip(p=1),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Base augmentation used for left upper corner cropping
base_augmentation_no_norm = transforms.Compose([
            transforms.Lambda(lambda img: img.crop((0, 0, 200, 200))),
            transforms.Resize((64, 64)),
            transforms.ToTensor(), 
            transforms.RandomHorizontalFlip(p=1),
            transforms.ColorJitter(brightness=.5, hue=.3),
            ])


def get_dataloaders(transform = base_transform_64, 
                    augmentation = base_augmentation,
                    path_first = './data/class1',
                      path_second = './data/class2', 
                      root_dir = "./data" , 
                      train_size = 0.7, val_size = 0.15, batch_size = 4, difference = 7500):
    
    '''Function that extracts train, validation and test sets'''

    first_class_paths = os.listdir(path_first)
    second_class_paths = os.listdir(path_second)

    size_from_first = int(len(first_class_paths)*0.5)
    size_from_second = int(len(second_class_paths)*0.5)

    indoor_outdoor_dataset = ImageFolder(root=root_dir, transform=transform)
    indoor_outdoor_dataset_augmented = ImageFolder(root=root_dir, transform=augmentation)

    indoor_dataset = Subset(indoor_outdoor_dataset, range(0, size_from_first))
    outdoor_dataset = Subset(indoor_outdoor_dataset, range(len(first_class_paths), len(first_class_paths) + size_from_second - 1))
    outdoor_dataset_augmented = Subset(indoor_outdoor_dataset_augmented,
                                       range(len(first_class_paths), len(first_class_paths) + difference))

    full_data = ConcatDataset([indoor_dataset, outdoor_dataset, outdoor_dataset_augmented])

    train_size = int(train_size * len(full_data))  # 70% for train
    val_size = int(val_size * len(full_data))   # 15% for validation
    test_size = len(full_data) - train_size - val_size  # Rest for the test

    train_dataset, val_dataset, test_dataset = random_split(full_data, [train_size, val_size, test_size], 
                                                            generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return train_loader, val_loader, test_loader

def compare_length_datasets(dataloader):

    '''Function that calcs number of labels presented in dataset'''
    
    labels_total = np.array([])

    for batch in tqdm(dataloader):

        images, labels = batch

        labels = labels.cpu().numpy()
        labels_total = np.append(labels_total, labels)
    
    return labels_total

def get_classes_info(array):
    '''Prints some info about data'''  

    print(f"Number of images in dataset: {len(array)}")
    print(f"Number of classes presented in data: {len(np.unique(array))}")

    print()
    
    mask_class_1 = np.where(array ==  0)
    mask_class_2 = np.where(array == 1)

    first_class = array[mask_class_1]
    second_class = array[mask_class_2]

    print(f"Number of images by class 0: {len(first_class)}")
    print(f"Number of images by class 1: {len(second_class)}")

    print()

    if len(first_class) == len(second_class):
        print("No disbalance in data")
        print()
    
    else:
        if len(first_class) > len(second_class):
            print("Disbalance: number in first is greater")
        else:
            print("Disbalance: number in second is greater")
        print(f"Difference: {abs(len(first_class) - len(second_class))}")

def print_info(train_loader, val_loader, test_loader):
    
    '''Prints the lengths of datasets'''

    print(f"Length of train_loader: {len(train_loader)}")
    print(f"Length of val_loader: {len(val_loader)}")
    print(f"length of test_loader: {len(test_loader)}")

def matplotlib_imshow(img, one_channel=False):
    
    '''Normalized image plotting function'''
    if one_channel:
        img = img.mean(dim=0)
    img = img * 0.5 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def save_checkpoint(state, filename='checkpoint.pth'):
    
    """Save model state"""

    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth'):
    
    '''Loads saved model'''
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict']) 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
    epoch = checkpoint['epoch'] 
    loss = checkpoint['loss'] 
    accuracy = checkpoint['accuracy']  
    print(f"Checkpoint loaded. Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}")
    return epoch, loss, accuracy


# Define class for classic transfrom model
class CNNIndoorOutdoorDoubleConvAugmentated(nn.Module):
    
    def __init__(self):
        super(CNNIndoorOutdoorDoubleConvAugmentated, self).__init__()
        self.convolution_layers = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fully_connected_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_image):
        x = self.convolution_layers(input_image)
        x = x.view(x.size(0), -1)  
        x = self.fully_connected_layers(x)
        
        return x

# Consts
modelCNNIndoorOutdoorDoubleConv = CNNIndoorOutdoorDoubleConvAugmentated()
optimizer = optim.Adam(modelCNNIndoorOutdoorDoubleConv.parameters(), lr=0.001)  
optimizer_adadelta = optim.Adadelta(modelCNNIndoorOutdoorDoubleConv.parameters(), lr=0.0005)
path_tensorboard = 'runs/running_indoor_outdoor_model_double_convolution'
writer = SummaryWriter(path_tensorboard)  

def evaluate_model(model, test_loader, criterion, device):
    
    '''Provides logs from epoch to epoch (predicting results on val set)'''

    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view(labels.size(0), -1))
            running_loss += loss
            predicted = (outputs.view(-1) > 0.5).long() 
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(test_loader)  
    accuracy = correct / total 
    
    print(f'Accuracy on test set: {accuracy:.4f}')
    print(f'Average Loss on test set: {avg_loss:.4f}')

    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, criterion, optimizer, file, device = 'cpu', num_epochs=10, 
                first_flag = True,
                optimizer_new_flag = False, optimizer_new = optimizer_adadelta):
    if first_flag:
        best_val_acc = 0.0
        epoch_last = 0
    else:
        epoch_last, loss, accuracy = load_checkpoint(model, optimizer, file)
        if optimizer_new_flag:
            optimizer = optimizer_new

        best_val_acc = accuracy
    
    model = model.to(device)

    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels.view(labels.size(0), -1))
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs.view(-1) > 0.5).long() 
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        
        writer.add_scalar('Loss/train', train_loss, epoch + epoch_last)
        writer.add_scalar('Accuracy/train', train_acc, epoch + epoch_last)
        writer.add_scalar('Loss/val', val_loss, epoch + epoch_last)
        writer.add_scalar('Accuracy/val', val_acc, epoch + epoch_last)
        

        if val_acc > best_val_acc:
            print(f'Validation accuracy improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model...')
            best_val_acc = val_acc
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_acc
            }
            save_checkpoint(checkpoint, file)

def predict_model(model, test_loader, device):
    
    '''Function to get all data we need'''

    model.eval()

    predicted_array = []
    labels_array = []
    with torch.no_grad():
        
        for i, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device).float().numpy()
            outputs = model(inputs)
            predicted = (outputs.view(-1) > 0.5).long().numpy()
            predicted_array.append(predicted)
            labels_array.append(labels)

    accuracy = accuracy_score(y_pred=np.concatenate(predicted_array), y_true=np.concatenate(labels_array))
    classification_report_task = classification_report(y_pred=np.concatenate(predicted_array), y_true=np.concatenate(labels_array))
    confusion_matrix_task = confusion_matrix(y_pred=np.concatenate(predicted_array), y_true=np.concatenate(labels_array))

    return classification_report_task, confusion_matrix_task, accuracy

def imshow(img, title):
    """image show function"""
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(title)
    plt.show()

criterion = nn.BCELoss()

def inference_and_visualize_errors(model, test_loader, device ='cpu'):
    incorrect_images = []
    incorrect_labels = []
    incorrect_preds = []
    losses = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device).float()

            outputs = model(inputs)
            
            predicted = (outputs > 0.5).float().view(-1)
            loss = criterion(outputs, labels.view(labels.size(0), -1))


            incorrect_indices = (predicted != labels).nonzero(as_tuple=True)[0]
            losses.append(loss.item())

            for idx in incorrect_indices:
                incorrect_images.append(inputs[idx].cpu())
                incorrect_labels.append(labels[idx].cpu().item())
                incorrect_preds.append(predicted[idx].cpu().item())
            

    for i in range(min(10, len(incorrect_images))):
        imshow(incorrect_images[i], f'Actual: {int(incorrect_labels[i])}, Predicted: {int(incorrect_preds[i])}')

        writer.add_image(f'Error_{i}', incorrect_images[i])
        writer.add_text(f'Error_{i}', f'Actual: {int(incorrect_labels[i])}, Predicted: {int(incorrect_preds[i])}')
    return losses


# Inheritated network to left upper corner approach
class CNNIndoorOutdoorDoubleConvAugmentatedCrop(CNNIndoorOutdoorDoubleConvAugmentated):
    
    def __init__(self):
        super(CNNIndoorOutdoorDoubleConvAugmentatedCrop, self).__init__()
       




