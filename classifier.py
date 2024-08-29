import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

from transformers import AutoProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
from dataset import *



device = 'cuda' if torch.cuda.is_available() else 'cpu'

import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, input_size=768, output=128):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers: 
        
        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in  ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels = None):
        '''
        input shape (N, in_features)
        '''

        
        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if labels is None:
            return wf

        numerator = self.s * torch.cos(torch.acos(torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1.+self.eps, 1-self.eps)) + self.m)
        
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)
    

class TESTMODEL(nn.Module):
    def __init__(self, in_feature=10, loss_type='arcface', arc_option=None):
        super(TESTMODEL, self).__init__()
        self.convlayers = FCNet(input_size=in_feature, output=arc_option['out_features']*2)
        self.adms_loss = AngularPenaltySMLoss(in_features=arc_option['out_features']*2, 
                                 out_features=arc_option['out_features'],
                                 s=arc_option['scaling_facor'],
                                 m=arc_option['margin'],
                                 loss_type=loss_type).to(device)

    def forward(self, x, labels=None, embed=False):
        x = self.convlayers(x)
        if embed:
            return x
        
        if labels is None:
            return self.adms_loss.predict(x)
        
        L = self.adms_loss(x, labels)
        return L

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

        
        self.data = []
        self.labels = []
        K, N, D = text_features.shape
        
        
        for k in range(K):
            for n in range(N):
                
                if Wrong_img is not None:
                    if k in Wrong_img[class_names[n]]:
                        continue
                
                self.feature.append(text_features[k, n])
                if image_features is not None :
                    self.feature.append(image_features[k, n])
                    self.labels.append(n)
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
    
    dataset = FeatureDataset(class_names = model.class_names,
                            text_features=t_feature,
                            image_features=i_feature,
                            Wrong_img=Wrong_img)
    #classifier = Classifier(num_classes=len(model.class_names)).to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    metric_fc = ArcMarginProduct(in_features=arc_option['in_features'], 
                                 out_features=arc_option['out_features'],
                                 s=arc_option['scaling_facor'],
                                 m=arc_option['margin']).to(device)
    
    c_optimizer = torch.optim.SGD(metric_fc.parameters(), lr=learning_rate, momentum=momentum)
    
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
            # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss : {round(loss.item(),4)}")


    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Trai ning ended at: {end_time.strftime('%H:%M:%S')}")
    hours, rem = divmod(duration.total_seconds(), 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total training time: {hours}h {mins}m {secs}s")
    
    torch.save(metric_fc.state_dict(), os.path.join(save_dir, 'models', f'arfFace1.pth'))
    
    # ArcFace loss 시각화
    plt.figure(figsize=(10, 6))
    
    plt.plot(ArcFace_Loss)
    plt.title('ArcFace Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'ArcFace Loss.png'))
    
    print("Dataset Len:", dataset.__len__(), "Total Len :" , model.K * len(model.class_names))
    

def train_classifier_NEW(t_feature, i_feature, model, arc_option, num_epochs, batch_size, learning_rate, momentum, save_dir, Wrong_img):
    
    dataset = FeatureDataset(class_names = model.class_names,
                            text_features=t_feature,
                            image_features=i_feature,
                            Wrong_img=Wrong_img)
    #classifier = Classifier(num_classes=len(model.class_names)).to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    metric_fc = AngularPenaltySMLoss(in_features=arc_option['in_features'], 
                                 out_features=arc_option['out_features'],
                                 s=arc_option['scaling_facor'],
                                 m=arc_option['margin']).to(device)
    
    c_optimizer = torch.optim.SGD(metric_fc.parameters(), lr=learning_rate, momentum=momentum)

    
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
            loss = metric_fc(feat, label)

            c_optimizer.zero_grad()
            loss.backward()
            c_optimizer.step()

            ArcFace_Loss.append(loss.item())
            # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss : {round(loss.item(),4)}  Batch len : {len(dataloader)}")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Trai ning ended at: {end_time.strftime('%H:%M:%S')}")
    hours, rem = divmod(duration.total_seconds(), 3600)
    mins, secs = divmod(rem, 60)
    print(f"Total training time: {hours}h {mins}m {secs}s")
    
    torch.save(metric_fc.state_dict(), os.path.join(save_dir, 'models', f'arfFace2.pth'))
    
    # ArcFace loss 시각화
    plt.figure(figsize=(10, 6))
    
    plt.plot(ArcFace_Loss)
    plt.title('ArcFace Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'ArcFace Loss.png'))
    
    print("Dataset Len:", dataset.__len__(), "Total Len :" , model.K * len(model.class_names))
    print("Classifier Training Completed")
      
      
def train_classifier_NEW_NEW(t_feature, i_feature, model, arc_option, num_epochs, batch_size, learning_rate, momentum, save_dir, Wrong_img):
    
    dataset = FeatureDataset(class_names = model.class_names,
                            text_features=t_feature,
                            image_features=i_feature,
                            Wrong_img=Wrong_img)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    
    classifier = TESTMODEL(in_feature=arc_option['in_features'],arc_option=arc_option).to(device)
    
    c_optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=momentum)

    
    start_time = datetime.now()
    ArcFace_Loss = []
    classifier.train()
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        for i, (feat, label) in enumerate(dataloader):

            # t_data = feat[0].to(device)
            # i_data = feat[1].to(device)
            feat = feat.to(device)
            label = label.to(device)

            # feat = torch.cat((t_data, i_data), dim=-1)
            loss = classifier(feat, label)

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
    
    torch.save(classifier.state_dict(), os.path.join(save_dir, 'models', f'arfFace.pth'))
    
    # ArcFace loss 시각화
    plt.figure(figsize=(10, 6))
    
    plt.plot(ArcFace_Loss)
    plt.title('ArcFace Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, 'ArcFace Loss.png'))
    
    #print("Dataset Len:", dataset.__len__(), "Total Len :" , model.K * len(model.class_names))
    print("Classifier Training Completed")
      
    
def inference_classifier(pth_path, dataset, config, classname_list, model, arc_option):
    metric_fc = ArcMarginProduct(in_features=arc_option['in_features'], 
                                 out_features=arc_option['out_features'],
                                 s=arc_option['scaling_facor'],
                                 m=arc_option['margin']).to(device)
    metric_fc.load_state_dict(torch.load(pth_path))
    if 'PACS' in dataset:
        val_data = load_PACS(16)
        label_name = 'labels'
        
    if 'OfficeHome' in dataset:
        val_data = load_OfficeHome(16, classname_list)
        label_name = 'labels'
        
    if 'VLCS' in dataset:
        val_data = load_VLCS(16)
        label_name = 'labels'

    processor = model.processor
    
    Total_img = 0
    Total_correct = 0
    import cv2
    for i, samples in enumerate(tqdm(val_data)):
        if 'PACS' in dataset:
            batch_images = [cv2.resize(sample['images'], (224,224), interpolation=cv2.INTER_LINEAR) for sample in samples]
            batch_labels = torch.tensor([sample[label_name].item() for sample in samples]).to(device)

        if 'VLCS' in dataset or 'OfficeHome' in dataset:
            batch_images, batch_labels = samples[0], samples[1].to(device)
        
        style_img_token = processor(images=batch_images, return_tensors="pt").to(device)

        style_img_feature = model.CLIP.get_image_features(**style_img_token)
        style_img_feature = style_img_feature / torch.norm(style_img_feature, p=2, dim=-1, keepdim=True)  # 정규화

        # prob = metric_fc(torch.concat([style_img_feature, style_img_feature], dim=-1))

        predict = metric_fc(style_img_feature)
        predict = torch.argmax(predict, dim=-1)

        Total_correct += (predict == batch_labels).sum().item()
        Total_img += len(batch_images)
        
        torch.cuda.empty_cache()
    print("Total Image : ", Total_img)
    print("Accuracy : ", round(Total_correct / Total_img * 100, 1), "%")
    


def inference_classifier_NEW(pth_path, dataset, config, classname_list, model, arc_option):
    metric_fc = AngularPenaltySMLoss(in_features=arc_option['in_features'], 
                                 out_features=arc_option['out_features'],
                                 s=arc_option['scaling_facor'],
                                 m=arc_option['margin']).to(device)
    metric_fc.load_state_dict(torch.load(pth_path))
        
    if 'PACS' in dataset:
        val_data = load_PACS(16)
        label_name = 'labels'
        
    if 'OfficeHome' in dataset:
        val_data = load_OfficeHome(16, classname_list)
        label_name = 'labels'
        
    if 'VLCS' in dataset:
        val_data = load_VLCS(16)
        label_name = 'labels'

    processor = model.processor
    
    Total_img = 0
    Total_correct = 0
    import cv2
    for i, samples in enumerate(tqdm(val_data)):
        if 'PACS' in dataset:
            batch_images = [cv2.resize(sample['images'], (224,224), interpolation=cv2.INTER_LINEAR) for sample in samples]
            batch_labels = torch.tensor([sample[label_name].item() for sample in samples]).to(device)

        if 'VLCS' in dataset or 'OfficeHome' in dataset:
            batch_images, batch_labels = samples[0], samples[1].to(device)
        
        style_img_token = processor(images = batch_images, return_tensors="pt",padding="max_length", truncation=True).to(device)

        style_img_feature = model.CLIP.get_image_features(pixel_values=style_img_token.pixel_values)

        # prob = metric_fc(torch.concat([style_img_feature, style_img_feature], dim=-1))

        predict = metric_fc(style_img_feature)
        
        predict = torch.argmax(predict, dim=-1)
        Total_correct += (predict == batch_labels).sum().item()
        Total_img += len(batch_images)
        
        torch.cuda.empty_cache()
    print("Total Image : ", Total_img)
    print("Accuracy : ", round(Total_correct / Total_img * 100, 1), "%")
    
    

def inference_classifier_NEW_NEW(pth_path, dataset, config, classname_list, model, arc_option):
    
    classifier = TESTMODEL(in_feature=arc_option['in_features'],arc_option=arc_option).to(device)

    classifier.load_state_dict(torch.load(pth_path))
    
    if 'PACS' in dataset:
        val_data = load_PACS(16)
        label_name = 'labels'
        
    if 'OfficeHome' in dataset:
        val_data = load_OfficeHome(16, classname_list)
        label_name = 'labels'
        
    if 'VLCS' in dataset:
        val_data = load_VLCS(16)
        label_name = 'labels'

    processor = AutoProcessor.from_pretrained(config[config['CLIP']]['tf'])
    
    Total_img = 0
    Total_correct = 0
    import cv2
    for i, samples in enumerate(tqdm(val_data)):
        if 'PACS' in dataset:
            batch_images = [cv2.resize(sample['images'], (224,224)) for sample in samples]
            batch_labels = torch.tensor([sample[label_name].item() for sample in samples]).to(device)

        if 'VLCS' in dataset or 'OfficeHome' in dataset:
            batch_images, batch_labels = samples[0], samples[1].to(device)
        
        style_img_token = processor(images=batch_images, return_tensors="pt").to(device)

        style_img_feature = model.CLIP.get_image_features(**style_img_token)
        style_img_feature = style_img_feature / torch.norm(style_img_feature, p=2, dim=-1, keepdim=True)  # 정규화

        # prob = metric_fc(torch.concat([style_img_feature, style_img_feature], dim=-1))

        predict = classifier(style_img_feature)

        Total_correct += (predict == batch_labels).sum().item()
        Total_img += len(batch_images)
        
        torch.cuda.empty_cache()
    print("Total Image : ", Total_img)
    print("Accuracy : ", round(Total_correct / Total_img * 100, 1), "%")
    
    
def text_transformer_forward( x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(model.transformer.resblocks):
            if i == len(model.transformer.resblocks):
                break
            if model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                pass
            else:
                x = r(x, attn_mask=attn_mask)
        return x
if __name__ == "__main__":
    
    import open_clip
    import torch
    import torch.nn as nn
    from diffusers import StableDiffusionPipeline

    model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', device='cuda')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')


    tokenizer = tokenizer
    processor = preprocess

    dim = 1024
    1

    text = tokenizer(["a photo of a dog"]).to(device)
    
    
    x = model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = text_transformer_forward(x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)
    
    print("text embedding : ", x.shape)
    
    diff = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker=None, torch_dtype=torch.float16).to(device)
    
    
'''
model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', device='cuda')
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    
    t = tokenizer(["a photo of a dog", "a photo of a cat"]).to(device)
    
    print("token : ", t.shape)
    
    cast_dtype = model.transformer.get_cast_dtype()
    x = model.token_embedding(t).to(cast_dtype)
    
    print("text embedding : ", x.shape)
    
    x = x + model.positional_embedding.to(cast_dtype)
    x = model.transformer(x, attn_mask = model.attn_mask)
    x = model.ln_final(x)
    
    print("final embedding : ", x.shape)
    
    x, _ = text_global_pool(x, t, model.text_pool_type)
    
    if isinstance(model.text_projection, nn.Linear):
        x = model.text_projection(x)
    else:
        x = x @ model.text_projection
        
    print("text projection : ", x.shape)

'''
    