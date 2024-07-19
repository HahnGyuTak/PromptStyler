import torch
import numpy as np
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def do_something():

    random.seed()
    np.random.seed()
    torch.manual_seed(random.randint(0, 100000))  # 원하는 방식으로 시드 변경
    torch.cuda.manual_seed_all(random.randint(0, 100000))
    
    
def get_cuda_info(INFO):
    
    if torch.cuda.is_available():
        current_device_idx = torch.cuda.current_device()
        current_device_name = torch.cuda.get_device_name(current_device_idx)
        
        INFO["Device"] = f'{current_device_name} {current_device_idx}'
    else:
        INFO["Device"] = "CPU"


if __name__ == '__main__':
    pass