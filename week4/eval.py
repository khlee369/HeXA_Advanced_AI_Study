import torch
from tqdm import tqdm

def accuracy(model, data_loader):
    correct = 0
    with torch.no_grad():
        for x,y in tqdm(data_loader):
            pred = model(x)
            correct += pred.argmax(dim=1).eq(y).sum().item()
    
    accuracy = 100 * correct / len(data_loader.dataset)
    # print(accuracy)
    
    return accuracy