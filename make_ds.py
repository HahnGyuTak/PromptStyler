
import os
import tarfile

# 데이터셋이 저장된 디렉토리 경로


def make_tar():
    directory = 'real_world'

    dataset_dir = f'result/OfficeHome_ViTL14/txt2img_res_f32'  # 여기에 실제 데이터셋 디렉토리 경로를 입력하세요.
    # 생성할 tar 파일 이름

    class_list = sorted(os.listdir(dataset_dir), key=lambda x : x.lower())

    n_split = 8

    split_list = [class_list[i::n_split] for i in range(n_split)]
    CNT = 0
    for i, l in enumerate(split_list):
        tar_file_name = f'{i+1:03}.tar'
        
        # tar 파일 생성
        with tarfile.open(f"tar/{tar_file_name}", 'w') as tar:
            # 디렉토리 내의 모든 파일을 순회
            
            for class_name in l:
                # 이미지와 텍스트 파일만 선택
                cnt = 0
                for filename in os.listdir(os.path.join(dataset_dir, class_name)):
                    if 'INSTANCE_IMGS' in filename:
                        continue
                    if filename.endswith('.png') or filename.endswith('.txt'):
                        file_path = os.path.join(dataset_dir, class_name, filename)
                        # tar 파일에 추가
                        tar.add(file_path, arcname=filename)
                        cnt += 1
                print("Added", cnt, "files from", class_name)
                CNT += cnt/2


        print(f"Created {tar_file_name} with {CNT} image-text pairs.")
    print(f"cp tar/*.tar /home/gue707/OPEN_CLIP_Training-Evaluation/src/dataset/promptstyler_officehome/")

if __name__ == "__main__":
    
    # dir_list = os.listdir('result/OfficeHome_ViTL14/txt2img_res_f32/')
    
    # for d in dir_list:
    #     # 각 디렉토리 내의 파일 개수가 400개가 아니면 파일 리스트 출력
    #     if len(os.listdir(f'result/OfficeHome_ViTL14/txt2img_res_f32/{d}')) != 400:
            
    #         for file_name in os.listdir(f'result/OfficeHome_ViTL14/txt2img_res_f32/{d}'):
    #             # file_name과 똑같은 txt파일에 "a photo of a {d}"를 적고 저장
    #             if file_name.endswith('.png'):
    #                 with open(f'result/OfficeHome_ViTL14/txt2img_res_f32/{d}/{file_name[:-4]}.txt', 'w') as f:
    #                     f.write(f'a photo of a {d}')
    
    make_tar()
    
    '''
    scp src/tar/*.tar gue707@163.152.162.236:~/OPEN_CLIP_Training-Evaluation/src/dataset/
    '''