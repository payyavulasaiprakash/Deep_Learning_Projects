import argparse
import cv2
import numpy as np
import torch
import os
from time import time
from glob import glob
from backbones import get_model
from sklearn.preprocessing import normalize
import concurrent
import torch.nn.functional as F

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

def face_pairs_fun(folder):#modfied code
    face_pairs = []
    all_dirs = glob(folder+'/*')
    all_dirs.sort()
    for f in all_dirs:
        images=os.listdir(f)
        # print(images)
        if len(images)==2:
            image_0=os.path.join(f,images[0])
            image_1=os.path.join(f,images[1])
            face_pairs.append([image_0,image_1])
    return face_pairs

@torch.no_grad()
def inference_for_embedding(net, name='r100', img=None):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    t1 = time()
    feat = net(img.to(device))
    t2 = time()
    return feat,t2-t1

def cosine_distance(vecs):
    x, y = vecs
    if device == 'cpu':
        x = torch.from_numpy(x).cpu().float()
        y = torch.from_numpy(y).cpu().float()
    elif device == 'cuda':
        x = x.float()
        y = y.float()
    # Check for NaN values in input vectors
    # if torch.isnan(x).any() or torch.isnan(y).any():
    #     raise ValueError("Input vectors contain NaN values.")
    # Normalize vectors
    x = F.normalize(x, p=2, dim=1)
    y = F.normalize(y, p=2, dim=1)

    # Compute cosine similarity
    dot_prod = torch.sum(x * y, dim=1)
    out = dot_prod.cpu().numpy() if device == 'cuda' else dot_prod.numpy()

    return out

   
def compare_faces(image_1_path, image_2_path, weight):
    name = 'r100'

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future1 = executor.submit(inference_for_embedding, weight, name, image_1_path)
        future2 = executor.submit(inference_for_embedding, weight, name, image_2_path)

        img1_emb, inf_time1 = future1.result()
        img2_emb, inf_time2 = future2.result()
        # print('inf_time1,inf_time2....',inf_time1,inf_time2)
        
    score = cosine_distance([img1_emb, img2_emb])[0]
    print('score', score)        
    return score,os.path.basename(os.path.dirname(image_1_path)),inf_time1 


def save_text(_file, content):
    with open(_file, mode='w+', encoding='utf-8') as f:
        f.write(content)


def compare(folder, out_file, model_name, weight):
    face_pairs = face_pairs_fun(folder)
    print('total face pairs found', len(face_pairs))
    detected_scores = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _imgs in face_pairs:
            try:
                image_1 = _imgs[0]
                image_2 = _imgs[1]
            except:
                continue
            future = executor.submit(compare_faces, image_1, image_2, weight)
            futures.append(future)  # Append a tuple (future, __folder) to the futures list
            # print('type........',concurrent.futures.as_completed(futures))
        for future in concurrent.futures.as_completed(futures):  
            try:
                # print(future.result())
                cosine, sub_folder_name, _time_compare = future.result()
                detected_scores.append([sub_folder_name, str(cosine), str(_time_compare)])
            except Exception as e:
                print(f"Error processing future for folder {sub_folder_name}: {e}")
    generate_csv(detected_scores, folder, out_file, model_name)


def generate_csv(detected, folder, out_file, model_name):
    folder_name = os.path.basename(folder)
    csv = 'folder,cosine similarity score, face comaparison time \n'
    for x in detected:
        _csv = ','.join(x)
        csv = csv + _csv + '\n'
    save_text('output_csvs_folders/{}/{}_{}.csv'.format(out_file,model_name,folder_name), csv)

def main(args):
    models_folder = args.models_folder
    test_data_folders = args.test_data_folders
    output_folder = args.output_folder
    test_during_training = args.test_during_training
    
    all_model_paths = sorted(glob('{}/*.pt'.format(models_folder)),reverse=True)
    if test_during_training:
        all_model_paths = [all_model_paths[0]]
    print(all_model_paths)
    if len(all_model_paths)>0:
        for weight in all_model_paths:
            model_name = os.path.basename(weight)
            name = 'vit_b_dp005_mask_005'  #'r100' 
            if device=='cuda':
                net = get_model(name, fp16=False).cuda()
            else:
                net = get_model(name, fp16=False)
            print("model_name",model_name)
            if 'check' not in weight:
                net.load_state_dict(torch.load(weight))
                model = torch.nn.DataParallel(net)
                model.eval()
            else:
                dict_checkpoint = torch.load(weight) #last checkpoint model when resuming
                net.load_state_dict(dict_checkpoint["state_dict_backbone"])
                model = torch.nn.DataParallel(net)
                model.eval()
                
            all_sub_folder_paths = [sub_folder.path for sub_folder in os.scandir(args.test_data_folders) if sub_folder.is_dir() ]
            for sub_folder_path in all_sub_folder_paths: 
                print(sub_folder_path)
                compare(sub_folder_path, output_folder, model_name,net)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Comparison eval')
    parser.add_argument('--data_folders', type=str, required=True,
                        help='main folder path, should consists of fake and real as sub folders')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--test_during_training', type=str, required=False, default=False,
                        hlp='test_during_training')
    parser.add_argument('--models', type=str, required=True, default='h.h5',
                        help='models folder path')
    args = parser.parse_args()
    main(args)


    

