import torch
from tqdm import tqdm

def evalualte(model, dataloader, epoch, total_epochs, device):    
    
    model.eval()  # Set the network to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{total_epochs}")
            for data in tepoch:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch+1}, Test Set Accuracy: {accuracy:.2f}%")
    
    return accuracy