import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import MNISTData
from model import MNISTNet

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tqdm import tqdm

from eval import accuracy

def main():
    train_data = MNISTData(dset='train')
    val_data = MNISTData(dset='val')

    train_loader = DataLoader(train_data, batch_size=8)
    val_loader = DataLoader(val_data, batch_size=8)

    loss_func = F.cross_entropy
    # loss_func

    Net = MNISTNet()

    optimizer = optim.Adam(Net.parameters(), lr=1e-3)

    Net.train()
    for epoch in range(1):
        for x, y in tqdm(train_loader):
            pred = Net(x)
            loss = loss_func(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(loss.item())

    Net.eval()
    print("TRAIN ACCURACY")
    print(accuracy(Net, train_loader))

    print("VAL ACCURACY")
    print(accuracy(Net, val_loader))

    torch.save(Net.state_dict(), './model.pt')

if __name__ == '__main__':
    main()