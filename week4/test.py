import torch

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

    Net = MNISTNet()
    Net.load_state_dict(torch.load('./model.pt'))
    Net.eval()

    print("TRAIN ACCURACY")
    print(accuracy(Net, train_loader))

    print("VAL ACCURACY")
    print(accuracy(Net, val_loader))

if __name__ == '__main__':
    main()