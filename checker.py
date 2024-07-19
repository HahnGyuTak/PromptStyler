import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import csv

def load_images_from_folder(folder):
    images = []
    filenames = []
    
    class_name = sorted(os.listdir(folder))
    for filename in class_name:
        if any(filename.lower().endswith(ext) for ext in ['jpg', 'jpeg', 'png']):
            img_path = os.path.join(folder, filename)
            images.append(Image.open(img_path))
            filenames.append(filename)
    return images, filenames

def classify_images(model, processor, images, class_names, device):
    inputs = processor(text=class_names, images=images, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return probs

def save_probabilities_to_csv(probabilities, filenames, class_names, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Filename'] + class_names
        writer.writerow(header)
        for i, filename in enumerate(filenames):
            row = [filename] + [round(prob, 3) for prob in probabilities[i].tolist()]
            writer.writerow(row)


def check(dataset, save_path):
    

    device =  "cpu" if torch.cuda.is_available() else "cpu"
    

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    result_dir = f"result/{dataset}/txt2img_res_f32"
    class_dirs = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d))]
    
    len_class = len(class_dirs)
    
    all_filenames = []
    all_probabilities = []
    
    percent = {}
    Wrong = {}
    correct = 0
    for class_dir in tqdm(class_dirs):
        class_path = os.path.join(result_dir, class_dir)
        images, filenames = load_images_from_folder(class_path)
        img_len = len(images)
        
        if not images:
            continue
        
        if device == "cuda":
            batch_size = 2
            split_imgs = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]
            
            probs = None
            for i, imgs in enumerate(images):
                prob = classify_images(model, processor, imgs, class_dirs, device)
                
                if probs is None:
                    probs = prob
                else:
                    probs = torch.cat((probs, prob), 0)
                
                torch.cuda.empty_cache()
                
            print(probs.shape)
        else :
            probs = classify_images(model, processor, images, class_dirs, device)
        cnt = 0
        
        wrong = []
        
        all_filenames.extend([os.path.join(class_dir, filename) for filename in filenames])
        all_probabilities.extend(probs.cpu().detach().numpy())
        
        for i, filename in enumerate(filenames):
            class_prob = probs[i]
            predicted_class_idx = class_prob.argmax().item()
            predicted_class = class_dirs[predicted_class_idx]
            is_correct = (predicted_class == class_dir)
            
            if is_correct:
                cnt += 1
            else:
                wrong.append(int(filename.split('.')[0].split('_')[1]))
            
            
            #print(f"Image: {filename} - Predicted: {predicted_class}, Actual: {class_dir}, Correct: {is_correct}")
        correct += cnt
        percent[class_dir] = cnt / img_len * 100
        Wrong[class_dir] = sorted(wrong)
        
    percent['total'] = correct / (img_len * len_class) * 100
    
    print(percent)
    

    with open(f'{save_path}/{dataset}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'percent'])
        for key in sorted(percent.keys()):
            if key != 'total':
                writer.writerow([key, percent[key]]+Wrong[key])
                #print([key, percent[key]]+Wrong[key])
        writer.writerow(['total', percent['total']])
    
    if not os.path.isdir('check'):
        os.mkdir('check')
    os.system(f'cp {save_path}/{dataset}.csv check/{dataset}.csv')
    
    return Wrong


if __name__ == "__main__":
    #check('Animal', "result/Animal")
    Wrong_img = {}
    with open('result/Animal/Animal.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i == 0 or row[0] == 'total':
                continue
            Wrong_img[row[0]] = list(map(int, row[2:]))
    print(Wrong_img)

