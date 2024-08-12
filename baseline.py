from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, AutoTokenizer, AutoProcessor
import torch
import cv2
from dataset import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference(clip, processor, data_loader, dataset, description = False):
    
    if dataset == 'PACS':
        class_names = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        
    if dataset == 'VLCS':
        class_names = ['bird', 'car', 'chair', 'dog', 'person']
    
    if dataset == 'OfficeHome':
        classname_dir = sorted(os.listdir('/home/dataset/OfficeHomeDataset_10072016/Art/'), key=str.lower)
        class_names = [name.replace(" ", "_") for name in classname_dir]
        
    if description:
        class_names = [f'a photo of a {name}' for name in class_names]
    
    Total_img = 0
    Total_correct = 0

    for i, samples in enumerate(tqdm(data_loader)):
        if 'PACS' in dataset:
            batch_images = [cv2.resize(sample['images'], (224,224), interpolation=cv2.INTER_LINEAR) for sample in samples]
            batch_labels = torch.tensor([sample['labels'].item() for sample in samples]).to(device)

        if 'VLCS' in dataset or 'OfficeHome' in dataset:
            batch_images, batch_labels = samples[0], samples[1].to(device)
        

        inputs = processor(text=class_names, 
                           images=batch_images, return_tensors="pt", padding=True).to(device)
        
        outputs = clip(**inputs)
        
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        predict = torch.argmax(probs, dim=-1)

        Total_correct += (predict == batch_labels).sum().item()
        Total_img += len(batch_images)
        
        torch.cuda.empty_cache()
    print("--------------------------------------------------")
    print("Dataset : ", dataset)
    print("Total Image : ", Total_img)
    print("Description : ", description ,"Accuracy : ", round(Total_correct / Total_img * 100, 1), "%")
    
    return round(Total_correct / Total_img * 100, 1)

def main():
    
    baseline_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # baseline_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")
    
    classname_dir = sorted(os.listdir('/home/dataset/OfficeHomeDataset_10072016/Art/'), key=str.lower)
    class_names = [name.replace(" ", "_") for name in classname_dir]
    
    pacs = load_PACS(16)
    vlcs = load_VLCS(16)
    officehome = load_OfficeHome(16,class_names)
    
    accuracy = {}
    accuracy['ZS-CLIP(C)'] = {}
    accuracy['ZS-CLIP(C)']['PACS'] = inference(baseline_clip, processor, pacs, 'PACS', False)
    accuracy['ZS-CLIP(C)']['VLCS'] = inference(baseline_clip, processor, vlcs, 'VLCS', False)
    accuracy['ZS-CLIP(C)']['OfficeHome'] = inference(baseline_clip, processor, officehome, 'OfficeHome', False)
    
    accuracy['ZS-CLIP(PC)'] = {}
    accuracy['ZS-CLIP(PC)']['PACS'] = inference(baseline_clip, processor, pacs, 'PACS', True)
    accuracy['ZS-CLIP(PC)']['VLCS'] = inference(baseline_clip, processor, vlcs, 'VLCS', True)
    accuracy['ZS-CLIP(PC)']['OfficeHome'] = inference(baseline_clip, processor, officehome, 'OfficeHome', True)
    
    print('ZS-CLIP(C) : ' ,accuracy['ZS-CLIP(C)'])
    print('ZS-CLIP(PC) : ' ,accuracy['ZS-CLIP(PC)'])
    
if __name__ == '__main__':
    main()