import deeplake
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
val_split = 0.2

class VLCSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.classname_list = sorted(['bird', 'car', 'chair', 'dog', 'person'])
        
        self.samples = self._make_dataset()
        np.random.shuffle(self.samples)

    def _make_dataset(self):
        samples = []
        for domain in tqdm(os.listdir(self.root_dir)):
            domain_path = os.path.join(self.root_dir, domain, 'test')
            if not os.path.isdir(domain_path):
                continue
            #for i, class_name in enumerate(self.classname_list):
            for i in range(5):
                class_path = os.path.join(domain_path, str(i))
                if not os.path.isdir(class_path):
                    continue
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            image = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
                            samples.append((image, i))
                        except (OSError, ValueError) as e:
                            print(f"Error loading image {img_path}: {e}")
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
    
        return image, label
        
        

class OfficeHomeDataset(Dataset):
    def __init__(self, root_dir, class_names,transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.classname_list = class_names
        
        self.samples = self._make_dataset()
        # sample 셔플
        np.random.shuffle(self.samples)

    def _make_dataset(self):
        samples = []
        for domain in tqdm(os.listdir(self.root_dir)):
            domain_path = os.path.join(self.root_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            
            for i, class_name in enumerate(self.classname_list):
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    print(f"Class {class_name} not found in {domain_path}")
                    continue
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            image = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
                            samples.append((image, i))
                        except (OSError, ValueError) as e:
                            print(f"Error loading image {img_path}: {e}")
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
    
        return image, label

def load_PACS(batch_size=32):
    ds = deeplake.load("hub://activeloop/pacs-test")
    
    pacs_loader = ds.dataloader().batch(batch_size) #.transform(transform=transform) # return samples as PIL images for transforms
 
    return pacs_loader
    
def load_OfficeHome(batch_size=32, class_names=['Art', 'Clipart', 'Product', 'RealWorld']):
   
    ds = OfficeHomeDataset(root_dir='/home/dataset/OfficeHomeDataset_10072016', class_names=class_names, transform=None)
    officehome_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return officehome_loader    

def load_VLCS(batch_size=32):
    
    ds = VLCSDataset(root_dir='/home/dataset/VLCS/VLCS_github', transform=None)
    vl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    print(len(vl) * batch_size)
    val_size = int(len(ds) * val_split)
    train_size = len(ds) - val_size
    _, val_ds = random_split(ds, [train_size, val_size])
    
    # DataLoader 생성
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    return vl 

if __name__ == "__main__":
    
    load_VLCS(1)
    
    

'''
Dataset/
        Domain1/
                Class1/
                        img1.jpg
                        img2.jpg
                        ...
                Class2/
                        img1.jpg
                        img2.jpg
                        ...
                ...
        Domain2/
                Class1/
                        img1.jpg
                        img2.jpg
                        ...
                Class2/
                        img1.jpg
                        img2.jpg
                        ...
                ...
        ...
        
'''