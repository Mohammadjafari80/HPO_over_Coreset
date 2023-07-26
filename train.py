import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm
from data import CustomWeightedDataset
from utils import WeightedCrossEntropyLoss, log_on_wandb
from eval import evalualte

def train_step(model, criterion, optimizer, dataloader, epoch, total_epochs, device):
    
    model.train()  # Set the network to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(dataloader, unit="batch") as t:
        t.set_description(f"Epoch {epoch+1}/{total_epochs}")

        for i, data in enumerate(t, 0):
            inputs, labels, w = data[0].to(device), data[1].to(device),  data[2].to(device)
            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels, w)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            t.set_postfix(loss=running_loss / (i + 1), accuracy = correct / total)

    
    return running_loss, correct / total
              

def train(model, train_dataset, test_dataset, weights, batch_size, device, total_epochs, milestone_ratios = [0.5, 0.75], test_interval = 5, learning_rate = 0.1, momentum = 0.9, weight_decay = 5e-4):
    
    weighted_dataset = CustomWeightedDataset(train_dataset, weights)
    train_loader = DataLoader(weighted_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # Create the test loader for evaluating test set accuracy
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define the loss function and optimizer
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    milestones = [int(r * total_epochs) for r in milestone_ratios]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    

    for epoch in range(total_epochs):
        results = {}
        loss, train_accuracy = train_step(model, criterion, optimizer, train_loader, epoch, total_epochs, device)
        results['train_loss'] = loss
        results['train_accuracy'] = train_accuracy
        
        scheduler.step()  # Update the learning rate

        if (epoch + 1) % test_interval == 0:
            test_accuracy = evalualte(model, test_loader, epoch, total_epochs, device)
            results['test_accuracy'] = test_accuracy
            
            
        log_on_wandb(results)