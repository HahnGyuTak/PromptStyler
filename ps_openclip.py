from diffusers import StableDiffusionPipeline

import torch
import torch.nn as nn

import numpy as np
from typing import Optional

import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class PromptStyler_OPENCLIP(nn.Module):
    def __init__(self , config, classes_list):

        super().__init__()

        self.N = len(classes_list)
        self.K = config['K']
        
        model, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K', device='cuda')
        tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    

        self.tokenizer = tokenizer
        self.processor = preprocess

        self.dim = config[config['CLIP']]['D']
        self.CLIP = model
        
        self.diff = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", safety_checker=None, torch_dtype=torch.float16).to(device)

        self.cast_dtype = self.CLIP.transformer.get_cast_dtype()
        pseudo_vector = torch.empty(self.K, 1, self.dim, dtype=self.cast_dtype)
        nn.init.normal_(pseudo_vector, mean=0, std=0.02)
        self.ctx = nn.Parameter(pseudo_vector) # [K, 1, D]

        # Style vector---------------------------------------
        style_word_vectors = torch.randn(self.K, 77, self.dim)

        p_style_list = ["a * style of a" for _ in style_word_vectors]

        token_style = self.tokenizer(p_style_list).to(device)
        self.token_p_style_list = token_style
        
        with torch.no_grad():
          self.embedding_p_style_list = self.CLIP.token_embedding(self.token_p_style_list).to(self.cast_dtype)
        
        self.p_style_start = self.embedding_p_style_list[:, :2, :] #[K, 2, D]
        self.p_style_end = self.embedding_p_style_list[:, 3:, :]    #[K, 74, D]



        # Contents vector---------------------------------------
        self.class_names = classes_list
        
        token_content = self.tokenizer(classes_list).to(device)
        
        self.token_p_content_list = token_content
        
        with torch.no_grad():
          self.embedding_p_content_list_tmp = self.CLIP.token_embedding(self.token_p_content_list).to(self.cast_dtype)
          
        x = self.embedding_p_content_list_tmp + self.CLIP.positional_embedding.to(self.cast_dtype)
        x = self.CLIP.transformer(x, attn_mask = self.CLIP.attn_mask)
        self.embedding_p_content_list_tmp = self.CLIP.ln_final(x)
        
        x, _ = text_global_pool(self.embedding_p_content_list_tmp, self.token_p_content_list, self.CLIP.text_pool_type)
        
        if isinstance(self.CLIP.text_projection, nn.Linear):
            x = self.CLIP.text_projection(x)
        else:
            x = x @ self.CLIP.text_projection
        
        self.content_feature = x / x.norm(dim=-1, keepdim=True, p=2) # [N, 512]


        # Style Conetent vector---------------------------------------
        prompt_list = ["a * style of a " + class_name for class_name in self.class_names]
        
        token_prompt = self.tokenizer(prompt_list).to(device)
        # token_prompt = self.tokenizer(prompt_list, return_tensors="pt",padding="max_length", truncation=True)
        
        self.token_prompt_list = token_prompt
        with torch.no_grad():
          self.embedding_prompt_list = self.CLIP.token_embedding(self.token_prompt_list).to(self.cast_dtype)
          
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
    
    def get_p_style_content_NEW_K(self, k):
        # ctx를 [K, N, 1, D] 형태로 확장
        ctx = self.ctx.unsqueeze(1).expand(self.K, self.N, 1, self.dim)[k] # [ N, 1, D]
        
        # prompt_start와 prompt_end를 [ N, ~, D] 형태로 확장
        prompt_start = self.prompt_start
        prompt_end = self.prompt_end
   
    
        # prompt_start, ctx, prompt_end를 concatenate하여 최종 [K, N, 77, D] 형태의 텐서 생성
        result = torch.cat([prompt_start, ctx, prompt_end], dim=1)
        
        return result


    def forward(self, k, p_case):
        # style
        if p_case == 'style':
            embedding_style = self.get_p_style() #[K, 77, D]
        
            x = embedding_style + self.CLIP.positional_embedding.to(self.cast_dtype)
            x = self.CLIP.transformer(x, attn_mask = self.CLIP.attn_mask)
            last_hidden_state = self.CLIP.ln_final(x)
            
            x, _ = text_global_pool(last_hidden_state, self.token_p_style_list, self.CLIP.text_pool_type)
            if isinstance(self.CLIP.text_projection, nn.Linear):
                x = self.CLIP.text_projection(x)
            else:
                x = x @ self.CLIP.text_projection
            
            style_feature = x / x.norm(dim=-1, keepdim=True, p=2) # [K, 1024]
            
            return [last_hidden_state, style_feature]

        #style-content
        if p_case == 'style_content':
            embedding_style_content = self.get_p_style_content_NEW_K(k) #[K, N, 77, D]
            
            x = embedding_style_content + self.CLIP.positional_embedding.to(self.cast_dtype)
            x = self.CLIP.transformer(x, attn_mask = self.CLIP.attn_mask)
            last_hidden_state = self.CLIP.ln_final(x)
            
            x, _ = text_global_pool(last_hidden_state, self.token_prompt_list, self.CLIP.text_pool_type)
            if isinstance(self.CLIP.text_projection, nn.Linear):
                x = self.CLIP.text_projection(x)
            else:
                x = x @ self.CLIP.text_projection
            
            style_content_feature = x / x.norm(dim=-1, keepdim=True, p=2) # [N, 1024]
            
            return [last_hidden_state, style_content_feature]





if __name__ == "__main__":

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
    
    
    
    
    