import os
import sys
import yaml
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from clip import clip
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, AutoTokenizer, AutoProcessor
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _create_4d_causal_attention_mask
from diffusers import StableDiffusionPipeline
import argparse
import matplotlib.pyplot as plt
from PIL import Image

from utils import *
from classifier import *
from checker import *

seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device)

class PromptStyler(nn.Module):
    def __init__(self , config, classes_list):

        super().__init__()

        self.N = len(classes_list)
        self.K = config['K']

        self.tokenizer = CLIPTokenizer.from_pretrained(config[config['CLIP']]['tf']) # Load CLIPTokenizer
        self.processor = AutoProcessor.from_pretrained(config[config['CLIP']]['tf']) # Load CLIPProcessor

        self.dim = config[config['CLIP']]['D']
        self.CLIP = CLIPModel.from_pretrained(config[config['CLIP']]['tf']).to(device)
        
        
        self.diff = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, torch_dtype=torch.float16).to(device)
        #self.diff_prompt_embedd = self.diff.encode_prompt

        dtype = self.CLIP.dtype
        pseudo_vector = torch.empty(self.K, 1, self.dim, dtype=dtype)
        nn.init.normal_(pseudo_vector, mean=0, std=0.02)
        self.ctx = nn.Parameter(pseudo_vector) # [K, 1, D]

        # Style vector---------------------------------------
        style_word_vectors = torch.randn(self.K, 77, self.dim)

        p_style_list = ["a * style of a" for _ in style_word_vectors]

        token_style = self.tokenizer(p_style_list, return_tensors="pt",padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True)
        # token_style = self.tokenizer(p_style_list, return_tensors="pt",padding="max_length", truncation=True).to(device)
        self.token_p_style_list = token_style['input_ids'].to(device)
        self.style_attention_mask = token_style.attention_mask.to(device)

        
        with torch.no_grad():
          self.embedding_p_style_list = self.CLIP.text_model.embeddings(input_ids = self.token_p_style_list.view(-1, self.token_p_style_list.size()[-1]))
          
        self.p_style_start = self.embedding_p_style_list[:, :2, :] #[K, 2, D]
        self.p_style_end = self.embedding_p_style_list[:, 3:, :]    #[K, 74, D]



        # Contents vector---------------------------------------
        self.class_names = classes_list
        
        token_content = self.tokenizer(classes_list, return_tensors="pt",padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True).to(device)
        # token_content = self.tokenizer(classes_list, return_tensors="pt",padding="max_length", truncation=True).to(device)
        
        self.token_p_content_list = token_content['input_ids'].to(device)
        
        with torch.no_grad():
          self.embedding_p_content_list_tmp = self.CLIP.text_model(input_ids = self.token_p_content_list, attention_mask=None)[0]
          
        last_hidden_state = self.embedding_p_content_list_tmp
        
        pooled_out = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                                      self.token_p_content_list.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                                      ].to(torch.float32)
        content_feature = self.CLIP.text_projection(pooled_out)
        self.content_feature = content_feature / content_feature.norm(p=2, dim=-1, keepdim=True)
        # print("[N, D] p_content feature shape :", self.content_feature.shape)
        # input()


        # Style Conetent vector---------------------------------------
        prompt_list = ["a * style of a " + class_name for class_name in self.class_names]
        
        token_prompt = self.tokenizer(prompt_list, return_tensors="pt",padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True)
        # token_prompt = self.tokenizer(prompt_list, return_tensors="pt",padding="max_length", truncation=True)
        
        self.token_prompt_list = token_prompt['input_ids'].to(device)
        self.prompt_attention_mask = token_prompt.attention_mask.to(device)
        with torch.no_grad():
          self.embedding_prompt_list = self.CLIP.text_model.embeddings(input_ids = self.token_prompt_list.view(-1, self.token_prompt_list.size()[-1]))
          
        print("------------------------------------------------------------------")
        # print("[N, 77, D] s-c feature embedding shape :", self.embedding_prompt_list.shape)

        self.prompt_start = self.embedding_prompt_list[:, :2, :] # [N, 2, D]
        self.prompt_end = self.embedding_prompt_list[:, 3:, :] # [N, 74, D]



    def get_p_style(self):
        p_style = torch.cat([self.p_style_start, self.ctx, self.p_style_end], dim=1)
        return p_style

    def get_p_style_content(self):
        ctx = self.ctx.expand([self.N, self.K, 1, self.dim])
        ctx = ctx.permute(1, 0, 2, 3) #[K, N, 1, 768]
        # ctx = self.ctx.unsqueeze(1)
        # ctx = ctx.repeat(1, self.N, 1, 1)

        return torch.stack([torch.cat([self.prompt_start, ctx[i], self.prompt_end], dim=1) for i in range(self.K)], dim=0)


    def get_p_style_content_NEW(self):
        # ctx를 [K, N, 1, D] 형태로 확장
        ctx = self.ctx.unsqueeze(1).expand(self.K, self.N, 1, self.dim)
        
        # prompt_start와 prompt_end를 [K, N, 1, D] 형태로 확장
        prompt_start = self.prompt_start.unsqueeze(0).expand(self.K, -1, -1, -1)
        prompt_end = self.prompt_end.unsqueeze(0).expand(self.K, -1, -1, -1)
        
        # prompt_start, ctx, prompt_end를 각각 [K, N, 77, D] 형태로 확장
        prompt_start = prompt_start.expand(-1, self.N, -1, -1)
        prompt_end = prompt_end.expand(-1, self.N, -1, -1)
        
        # prompt_start, ctx, prompt_end를 concatenate하여 최종 [K, N, 77, D] 형태의 텐서 생성
        result = torch.cat([prompt_start, ctx, prompt_end], dim=2)
        
        return result


    def forward(self, k, p_case):
        # style
        if p_case == 'style':
            embedding_style = self.get_p_style() #[K, 77, D]
        
            #style_feature = torch.cat([self.T_encoder(embedding_style[i].unsqueeze(0), self.token_p_style_list[i].unsqueeze(0)) for i in range(k)], dim=0) #[k, 512]
            attention_mask = None#_prepare_4d_attention_mask(self.style_attention_mask, embedding_style.dtype)
            causal_attention_mask = _create_4d_causal_attention_mask(
                  self.token_p_style_list.size(), embedding_style.dtype, device=embedding_style.device
              )
            last_hidden_state = self.CLIP.text_model.final_layer_norm(self.CLIP.text_model.encoder(inputs_embeds=embedding_style, attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0])
            
            #last_hidden_state = self.diff_prompt_embedd(prompt = None, device = device, num_images_per_prompt = 1, do_classifier_free_guidance = False, prompt_embeds = last_hidden_state)[0]
            
            #print("[k, 77, D] Embedding P_style", last_hidden_state.shape)
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), 
                                              self.token_p_style_list.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                                              ].to(torch.float32)
            #print("P_style", pooled_output.shape)
            style_feature = self.CLIP.text_projection(pooled_output)
            #print("T_style" ,style_feature.shape)
            style_feature = style_feature / style_feature.norm(dim=-1, keepdim=True, p=2) # [k, 512]
            
            return [last_hidden_state, style_feature]

        #style-content
        if p_case == 'style_content':
            embedding_style_content = self.get_p_style_content_NEW() #[K, N, 77, D]
            
            
            # style_content_feature = torch.cat([self.T_encoder(embedding_style_content[k][n].unsqueeze(0), self.token_prompt_list[n].unsqueeze(0)) for n in range(self.N)], dim=0) #[N, 512]
            attention_mask = None#_prepare_4d_attention_mask(self.prompt_attention_mask, embedding_style_content.dtype)
            
            causal_attention_mask = _create_4d_causal_attention_mask(
                  self.token_prompt_list.size(), embedding_style_content.dtype, device=embedding_style_content.device
              )
            last_hidden_state = self.CLIP.text_model.final_layer_norm(self.CLIP.text_model.encoder(inputs_embeds=embedding_style_content[k], attention_mask=attention_mask, causal_attention_mask=causal_attention_mask)[0])
            
            #last_hidden_state = self.diff_prompt_embedd(prompt = None, device = device, num_images_per_prompt = 1, do_classifier_free_guidance = False, prompt_embeds = last_hidden_state)[0]
            
            #print("[N, 77, D] Embedding P_style_content", last_hidden_state.shape)
            pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                                              self.token_prompt_list.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
                                              ].to(torch.float32)
            #print("P_style_content", pooled_output.shape)
            style_content_feature = self.CLIP.text_projection(pooled_output)
            #print("T_style_content" ,style_content_feature.shape)
            style_content_feature = style_content_feature / style_content_feature.norm(p=2, dim=-1, keepdim=True) #[N, D]

            return [last_hidden_state, style_content_feature]


def style_diveristy_loss(T_style, i, cos_sim):

    # L_style = 0
    # for j in range(i):

    #     L_style += abs(cos_sim(T_style[i], T_style[j]))

    # return L_style/i
     # 현재 벡터와 이전 벡터들 간의 코사인 유사도를 계산
    cos_sims = cos_sim(T_style[i], T_style[:i])
    
    # 코사인 유사도의 절대값의 평균을 계산
    L_style = torch.mean(torch.abs(cos_sims))
    
    return L_style


def content_consistency_loss(T_style_contents, T_contents, N, cos_sim):

    def exp_z_i(m, n):
        return torch.exp(cos_sim(T_style_contents[m], T_contents[n]))

    L_content = 0
    for m in range(N):
        L_content += torch.log(exp_z_i(m,m) / torch.sum(torch.stack([exp_z_i(m, n) for n in range(N) if n != m])))

    return -L_content / N#2 - math.log(N - 1 + math.exp(2)) - (L_content / N)


def content_consistency_loss_NEW(T_style_contents, T_contents, N, ce_loss):
  
  logits = torch.matmul(T_style_contents, T_contents.t())
  
  return ce_loss(logits, torch.arange(N).to(device))


if __name__ == '__main__':
    seed_everything(seed)
  
    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--data_dir', type=str, default='/home/dataset/OfficeHomeDataset_10072016/Art/')
    parser.add_argument('--save_dir', type=str, default='/home/gue707/PromptStyler/result/')
    parser.add_argument('--dataset', type=str, default='OfficeHome')
    parser.add_argument('--train', type=str, default='None', choices=['both', 'style', 'classifier', None],
        help='Choose one of the following: both, diff, classifier')
    parser.add_argument('--config', type=str, default="config/config_vitl14.yaml")
    parser.add_argument('--infer', type=str, default='diff', choices=['both', 'diff', 'classifier'],
        help='Choose one of the following: both, diff, classifier')
    
    args = parser.parse_args()

    classname_dir = sorted(os.listdir(args.data_dir), key=str.lower)
    classname_list = [name.replace(" ", "_") for name in classname_dir]

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if 'VLCS' in args.dataset:
        classname_list = sorted(['bird', 'car', 'chair', 'dog', 'person'])
    if 'PACS' in args.dataset:
        classname_list = sorted(['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person'])
    if 'Animal' in args.dataset:
        classname_list = sorted(['bird', 'cat', 'dog', 'fish', 'horse', 'rabbit', 'sheep', 'turtle', 'whale', 'elephant', 'lion', 'penguin', 'tiger'])
    
    
    print("Class N : ", len(classname_list))
    
    arc_options = {'in_features': config[config['CLIP']]['D'],
                    'out_features': len(classname_list),
                    'scaling_facor': config['train_cf']['scaling_facor'],
                    'margin': config['train_cf']['angular_margin']}
    
    save_dir = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'features'), exist_ok=True)
    
    Wrong_img = None
    
    K, L = config['K'], config['train_style']['epoch']
    
    if args.train == 'both' or args.train == 'style':
        
        train_INFO = {'K': K, 'epoch_per_style': L, 'N': len(classname_list)}
        get_cuda_info(train_INFO)
        
        model = PromptStyler(config = config, classes_list=classname_list).to(device)
        
        for name, param in model.named_parameters():
            if "ctx" not in name:
                param.requires_grad_(False) 
            
        
        cos_sim = nn.CosineSimilarity(dim=-1).to(device)
        
        learning_rate, momentum = config['train_style']['lr'], config['train_style']['momentum']
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        T_contents = model.content_feature.detach()
        Embed_contents = model.embedding_p_content_list_tmp.cpu()

        torch.save(T_contents, os.path.join(save_dir, 'features', f'contents_feat.pth'))
        #torch.save(model.embedding_p_content_list_tmp, os.path.join(save_dir, f'contents_emb_tmp.pth'))
        torch.save(Embed_contents, os.path.join(save_dir, 'features', f'contents_emb.pth'))
        
        ce_loss = nn.functional.cross_entropy
        
        Style_loss = {}
        Content_loss = {} 
        model.train()
        
        for i in range(1,  K):
            Style_loss[i] = []
            Content_loss[i] = []
            for iter in tqdm(range(L)):
                _, T_style = model(None, 'style')  # [i+1, 768]
                
                #Style Diversity Loss
                L_style = style_diveristy_loss(T_style, i, cos_sim)

                #Content Consistency Loss
                _, T_style_contents = model(i, 'style_content') #[N, 768]
                #L_content = content_consistency_loss_TEST(T_style_contents, T_contents, model.N, cos_sim)
                L_content = content_consistency_loss_NEW(T_style_contents, T_contents, model.N, ce_loss)
       
                # # Total Loss
                L_prompt = L_content + L_style
                
                # Gradient update
                optimizer.zero_grad()
                L_prompt.backward()
                optimizer.step()
                
                Style_loss[i].append(L_style.item())
                Content_loss[i].append(L_content.item())
                
                if (iter+1) % 10 == 0:
                  print(f"------------ [{i}/{K}][{iter+1}/{L}] -- L_style : {L_style :.4f}, L_content : {L_content:.4f}")

        with torch.no_grad():
            # prmopt_emb = torch.stack([model(K, 'style')[0] for ps in range(80)])
            # prmopt_feat = torch.stack([model(i+1, 'style')[1] for ps in range(80)])
            
            prmopt_emb = torch.cat([model(i, 'style_content')[0].unsqueeze(0) for i in range(K)], dim = 0)
            prompt_feat = torch.cat([model(i, 'style_content')[1].unsqueeze(0) for i in range(K)], dim = 0)
  
            # torch.save(prmopt_feat, os.path.join(save_dir, f'prompt_feat.pth'))
            torch.save(prmopt_emb.detach().cpu(), os.path.join(save_dir, 'features', f'prompt_emb.pth'))
            torch.save(prompt_feat.detach().cpu(), os.path.join(save_dir, 'features', f'prompt_feat.pth'))
            
            torch.save(model.ctx, os.path.join(save_dir, 'models', f'ctx.pth'))
            
        plt.figure(figsize=(10, 6))
        for i in range(1, model.K):
            plt.plot(Style_loss[i], label=f'Style {i}')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Style loss')
        plt.savefig(f'{save_dir}/Style Loss.png')

        plt.figure(figsize=(10, 6))
        for i in range(1, model.K):
            plt.plot(Content_loss[i], label=f'Style {i}')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Content Loss')
        plt.savefig(f'{save_dir}/Content Loss.png')
        
    
    if args.infer == 'both' or args.infer == 'diff':
        ### text_emb to image ###
      
        if args.train != 'style' and args.train != 'both':
            model = PromptStyler(config = config, classes_list=classname_list).to(device)
      
        image_result_dir = os.path.join(save_dir,"txt2img_res_f32")
        pipe = model.diff
        print(torch.load(os.path.join(save_dir, 'features', f'prompt_emb.pth')).shape)
        for m in range(len(classname_list)):
            for i in range(K):
                #style_emb_im = torch.load(os.path.join(save_dir, f'prompt_emb.pth'))[i][m] #torch.Size([n_cls, 77, 768])
                style_emb_im = torch.load(os.path.join(save_dir, 'features', f'prompt_emb.pth'))[i][m] #torch.Size([n_cls, 77, 768])
                image = pipe(prompt_embeds=style_emb_im.unsqueeze(0)).images[0]
                os.makedirs(os.path.join(image_result_dir, classname_list[m]), exist_ok=True) 
                image.save(f'{image_result_dir}/{classname_list[m]}/style_{i}.png')
                print(f"Save Image of {m}th Content({classname_list[m]}) of {i}th Style")

        Wrong_img = check(args.dataset, save_dir)

    if args.train == 'both' or args.train == 'classifier':
      
        # Classifierㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
        if args.train == 'classifier' and args.infer == 'classifier':
            prompt_feat = torch.load(os.path.join(save_dir, 'features', f'prompt_feat.pth')).cpu()
            model = PromptStyler(config = config, classes_list=classname_list).to(device)

        if "ViTL14" in args.dataset and Wrong_img is None:
            Wrong_img = {}
            with open(f'result/{args.dataset}/{args.dataset}.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for i, row in enumerate(reader):
                    if i == 0 or row[0] == 'total':
                        continue
                    Wrong_img[row[0]] = list(map(int, row[2:]))
        
        if "ViTL14" in args.dataset :
            if not os.path.exists(os.path.join(save_dir, 'features', f'image_feat.pth')):
                img_to_feature(imgs_dir = os.path.join(save_dir, 'txt2img_res_f32'),
                            save_dir = os.path.join(save_dir, 'features'))
                
                print("class_name :", classname_list)

            image_feat = torch.load(os.path.join(save_dir, 'features', f'image_feat.pth')).cpu()
            print("image_feat shape :", image_feat.shape)
            image_feat = None
            Wrong_img = None
            
        else :
            image_feat = None
            Wrong_img = None
        # image_feat = None
       # 
        print("prompt_feat shape :", prompt_feat.shape)
        
        train_classifier(t_feature = prompt_feat,
                         i_feature = image_feat,
                         model = model,
                         arc_option=arc_options,
                         num_epochs=config['train_cf']['epoch'],
                         batch_size=config['train_cf']['batch_size'],
                         learning_rate=config['train_cf']['lr'],
                         momentum=config['train_cf']['momentum'],
                         save_dir = save_dir,
                         Wrong_img = Wrong_img)
        
      
    if args.infer == 'both' or args.infer == 'classifier':
      
        if args.train != 'classifier' and args.train != 'both':
            model = PromptStyler(config = config, classes_list=classname_list).to(device)
        torch.cuda.empty_cache()

        path = os.path.join(save_dir, 'models', f'arfFace.pth')
        
        inference_classifier(path, args.dataset, config, classname_list, model, arc_options)
        
        
    
