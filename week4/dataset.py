from pathlib import Path
import requests
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle
import gzip

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

class MNISTData(Dataset):
    def __init__(self, dset='train'):
        if dset=='train':
            self.x = x_train
            self.y = y_train
        elif dset=='val':
            self.x = x_valid
            self.y = y_valid
        else:
            raise Exception('dset 정의 제대로 하셈 train or val')
            
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    train_data = MNISTData(dset='train')
    x, y = train_data[40]
    print(x.shape)
    print(y)