

officehome_classnames = ['alarm clock','backpack','batteries','bed','bike','bottle','bucket','calculator','calendar','candles',
'chair','clipboards','computer','couch','curtains','desk lamp','drill','eraser','exit sign','fan',
'file cabinet','flipflops','flowers','folder','fork','glasses','hammer','helmet','kettle','keyboard',
'knives','lamp shade','laptop','marker','monitor','mop','mouse','mug','notebook','oven',
'pan','paper clip','pen','pencil','postit notes','printer','push pin','radio','refrigerator','ruler',
'scissors','screwdriver','shelf','sink','sneakers','soda','speaker','spoon','tv','table',
'telephone','toothbrush','toys','trash can','webcam']

vlcs_classnames = ['bird','car','chair','dog','person']

pacs_classnames=  ['dog','elephant','giraffe','guitar','horse','house','person']

domainnet_classnames = []

downstream_template = [
 lambda c: f'a photo of a {c}'
# lambda c: f'{c}'
]
# lambda c: f'a photo of a {c}'
# lambda c: f'{c}'

'''
CUDA_VISIBLE_DEVICES=7

OFFICEHOME
CUDA_VISIBLE_DEVICES=7 python3 -m training.main \
  --zeroshot-frequency 1 \
  --batch-size=512 \
  --workers=16 \
  --model ViT-L-14 \
  --pretrained openai \
  --down-eval=/home/dataset/office-home/Art
  
PACS
CUDA_VISIBLE_DEVICES=7 python3 -m training.main \
--zeroshot-frequency 1 \
--batch-size=256 \
--workers=16 \
--model ViT-B-16 \
--pretrained openai \
--down-eval=data/pacs/art_painting
   
  
VLCS
CUDA_VISIBLE_DEVICES=7 python3 -m training.main \
--zeroshot-frequency 1 \
--batch-size=256 \
--workers=16 \
--model ViT-B-16 \
--pretrained openai \
--down-eval=data/vlcs/CALTECH/test

CUDA_VISIBLE_DEVICES=4 python3 -m training.main \
--zeroshot-frequency 1 \
--batch-size=256 \
--workers=16 \
--model ViT-L-14 \
--pretrained openai \
--down-eval=data/vlcs/CALTECH/test

'''

'''
CUDA_VISIBLE_DEVICES=3 python3 -m training.main \
--save-frequency 1 \
--zeroshot-frequency 1 \
--lr=1e-3 \
--wd=0.1 \
--epochs=30 \
--warmup 10000 \
--batch-size=256 \
--workers=16 \
--pretrained openai \
--model ViT-L-14 \
--down-eval=/home/gue707/dreembooth/diffusers/examples/dreambooth/OfficeHome/art \
--dataset_type auto \
--train_data office-home


'''