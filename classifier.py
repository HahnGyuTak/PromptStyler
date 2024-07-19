import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import math
import matplotlib.pyplot as plt

from transformers import AutoProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label = None):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        if label is None:
            return cosine
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class FeatureDataset(Dataset):
    def __init__(self, text_features, image_features, class_names, Wrong_img = None):
        self.feature = []
        #self.image_features = []

        
        self.data = []
        self.labels = []
        K, N, D = text_features.shape
        
        
        for k in range(K):
            for n in range(N):
                
                if Wrong_img is not None:
                    if k in Wrong_img[class_names[n]]:
                        continue
                
                self.feature.append(text_features[k, n])
                # self.feature.append(image_features[k, n])
                # self.labels.append(n)
                self.labels.append(n)
                
                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.feature[idx]
        label = self.labels[idx]
        
        return feature, label
                
class ConcatFeatureDataset(Dataset):
    def __init__(self, text_features, image_features, class_names, Wrong_img = None):
        self.text_feature = []
        self.image_features = []

        
        self.data = []
        self.labels = []
        K, N, D = text_features.shape
        
        
        for k in range(K):
            for n in range(N):
                
                if Wrong_img is not None:
                    if k in Wrong_img[class_names[n]]:
                        continue
                
                self.text_feature.append(text_features[k, n])
                self.image_features.append(image_features[k, n])
                self.labels.append(n)

                
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_feature = self.text_feature[idx]
        image_feature = self.image_features[idx]
        label = self.labels[idx]
        
        return (text_feature, image_feature), label

def img_to_feature(imgs_dir, save_dir, batch_size=8):  # 배치 크기를 추가 인자로 받음
    
    def extract_number(filename):
        return int(filename.split('_')[1].split('.')[0])
    
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    CLIP = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)  # CLIP 모델을 GPU로 이동
    
    class_names = sorted(os.listdir(imgs_dir))
    
    styles = sorted(os.listdir(os.path.join(imgs_dir, class_names[0])), key=extract_number)
    
    images = []
    for style in tqdm(styles):
        style_images = []
        for class_name in class_names:
            image_path = os.path.join(imgs_dir, class_name, style)
            image = Image.open(image_path)
            style_images.append(np.array(image))
        
        # 배치 크기만큼 이미지를 나눠서 처리
        batched_images = None
        for i in range(0, len(style_images), batch_size):
            batch_images = style_images[i:i + batch_size]
            style_img_token = processor(images=batch_images, return_tensors="pt").to(device) # 입력 데이터를 GPU로 이동
            
            with torch.no_grad():  # 메모리 사용 최적화를 위해 no_grad 사용
                style_img_feature = CLIP.get_image_features(**style_img_token)
                style_img_feature = style_img_feature / torch.norm(style_img_feature, p=2, dim=-1, keepdim=True)  # 정규화
            
            if batched_images is None:
                batched_images = style_img_feature
            else:
                batched_images = torch.cat([batched_images, style_img_feature], dim=0)

            torch.cuda.empty_cache()  # GPU 메모리 캐시를 비웁니다.
        images.append(batched_images.cpu())  # CPU로 이동한 뒤 images에 추가
    image_features = torch.stack(images)

    torch.save(image_features, os.path.join(save_dir, 'image_feat.pth'))
    print(image_features.shape)


def train_classifier(t_feature, i_feature, model, arc_option, num_epochs, batch_size, learning_rate, momentum, save_dir, Wrong_img):
    
    
    #classifier = Classifier(num_classes=len(model.class_names)).to(device)
    
    metric_fc = ArcMarginProduct(in_features=arc_option['in_features'], 
                                 out_features=arc_option['out_features'],
                                 s=arc_option['scaling_facor'],
                                 m=arc_option['margin']).to(device)
    
    c_optimizer = torch.optim.SGD(metric_fc.parameters(), lr=learning_rate, momentum=momentum)
    
    dataset = FeatureDataset(class_names = model.class_names,
                             text_features=t_feature,
                             image_features=i_feature,
                             Wrong_img=Wrong_img)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    start_time = datetime.now()
    ArcFace_Loss = []
    metric_fc.train()
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        for i, (feat, label) in enumerate(dataloader):

            # t_data = feat[0].to(device)
            # i_data = feat[1].to(device)
            feat = feat.to(device)
            label = label.to(device)

            # feat = torch.cat((t_data, i_data), dim=-1)
            output = metric_fc(feat, label)
            loss = criterion(output, label)

            c_optimizer.zero_grad()
            loss.backward()
            c_optimizer.step()

            ArcFace_Loss.append(loss.item())
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
        print(f"Epoch [{epoch+1}/{num_epochs}] Duration: {epoch_duration}s")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Trai ning ended at: {end_time.strftime('%H:%M:%S')}")
    hours, rem = divmod(duration.total_seconds(), 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total training time: {hours}h {mins}m {secs}s")
    
    torch.save(metric_fc.state_dict(), os.path.join(save_dir, 'models', f'arfFace.pth'))
    
    # ArcFace loss 시각화
    plt.figure(figsize=(10, 6))
    
    plt.plot(ArcFace_Loss)
    plt.title('ArcFace Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'ArcFace Loss.png'))
    
    print("Dataset Len:", dataset.__len__(), "Total Len :" , model.K * len(model.class_names))
    
    
    
def inference_classifier(metric_fc, model, image_path_list):
    
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    
    images = []
    for img_path in image_path_list:
        img = processor(images=Image.open(img_path), return_tensors="pt")
        image_features = model.CLIP.get_image_features(**img.to(device))
        image_features = image_features / torch.norm(image_features,p=2, dim=-1, keepdim=True)

        images.append(image_features)
        torch.cuda.empty_cache()
    image_features = torch.stack(images)

    
    # feature = metric_fc(torch.concat([image_features, image_features], dim = -1))
    feature = metric_fc(image_features)
    
    return feature
    

if __name__ == "__main__":
    
    feature = torch.load('result/PACS/prompt_feat.pth')
    classes = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
    
    dataset = FeatureDataset(feature, classes)
    
    data = []
    labels = []
    K, N, D = feature.shape
    
    le = LabelEncoder()
    
    encoded = le.fit_transform(classes)
    emb = torch.load('result/PACS/prompt_emb.pth')
    for k in range(K):
        for n in range(N):
            data.append(emb[k, n])
            labels.append((encoded[n], classes[n]))
 
    image_result_dir = "OfficeHome_diffuser/PACS"
    
    
    from diffusers import StableDiffusionPipeline
    
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    os.makedirs(image_result_dir, exist_ok=True)
    print(len(data))
    for i in range(10):
        emb = data[i].unsqueeze(0)
        
        img = pipe(prompt_embeds = emb).images[0]
        img.save(os.path.join(image_result_dir, f"{i}_{labels[i]}.png"))
    