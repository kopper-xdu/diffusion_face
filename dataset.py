from torch.utils.data import Dataset
import os
from PIL import Image


class base_dataset(Dataset):
    def __init__(self, path, transform=None) -> None:
        super().__init__()
        
        self.path = path
        self.img_names = sorted(os.listdir(path))
        self.transform = transform
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.img_names[index])
        img = Image.open(img_path).convert('RGB')
        tgt_img = Image.open('./17082.png').convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            tgt_img = self.transform(tgt_img)
            
        return img, tgt_img