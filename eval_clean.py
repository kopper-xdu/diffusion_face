import torch
from torch import nn
import numpy as np
import random

import os
from fr_model import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1
from dataset import base_dataset
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Subset
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_model(name):
    if name == 'IRSE50':
        model = IRSE_50()
        model.load_state_dict(torch.load('pretrained_model/irse50.pth', map_location=torch.device('cpu')))
    elif name == 'MobileFace':
        model = MobileFaceNet(512)
        model.load_state_dict(torch.load('pretrained_model/mobile_face.pth', map_location=torch.device('cpu')))
    elif name == 'IR152':
        model = IR_152([112, 112])
        model.load_state_dict(torch.load('pretrained_model/ir152.pth', map_location=torch.device('cpu')))
    elif name == 'FaceNet':
        model = InceptionResnetV1(num_classes=8631)
        model.load_state_dict(torch.load('pretrained_model/facenet.pth', map_location=torch.device('cpu')))
    else:
        raise ValueError(f'Invalid model name: {name}')
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    return model.cuda()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def main():
    seed = 0
    batch_size = 2
    num_workers = 0
    
    setup_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = base_dataset(path='./data', transform=transform)
    # dataset = Subset(dataset, [x for x in range(50)])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    attack_model_names = ['IR152']
    attack_model_dict = {'IR152': get_model('IR152'), 'IRSE50': get_model('IRSE50'), 
                         'FaceNet': get_model('FaceNet'), 'MobileFace': get_model('MobileFace')}
    # attack_model_resize_dict = {'IR152': 112, 'IRSE50': 112, 'FaceNet': 160, 'MobileFace': 112}
    # cos_sim_scores_dict = {'IR152': [], 'IRSE50': [], 'FaceNet': [], 'MobileFace': []}
    cos_sim_scores_dict = {'IR152': []}
    
    for attack_model_name in attack_model_names:
        attack_model = attack_model_dict[attack_model_name]
        resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d((160, 160))
        for i, (image, tgt_image) in enumerate(tqdm(dataloader)):
            tgt_image = tgt_image.to(device)
            image = image.to(device)
            B = image.shape[0]
            
            feature1 = attack_model(resize(image)).reshape(B, -1)
            feature2 = attack_model(resize(tgt_image)).reshape(B, -1)
                
            score = F.cosine_similarity(feature1, feature2)
            cos_sim_scores_dict[attack_model_name] += score.tolist()

    asr_calculation(cos_sim_scores_dict)


th_dict = {'IR152':(0.094632, 0.166788, 0.227922), 'IRSE50':(0.144840, 0.241045, 0.312703),
           'FaceNet':(0.256587, 0.409131, 0.591191), 'MobileFace':(0.183635, 0.301611, 0.380878)}

def asr_calculation(cos_sim_scores_dict):
    # Iterate each image pair's simi-score from "simi_scores_dict" and compute the attacking success rate
    for key, values in cos_sim_scores_dict.items():
        th01, th001, th0001 = th_dict[key]
        total = len(values)
        success01 = 0
        success001 = 0
        success0001 = 0
        for v in values:
            if v > th01:
                success01 += 1
            if v > th001:
                success001 += 1
            if v > th0001:
                success0001 += 1
        print(key, " attack success(far@0.1) rate: ", success01 / total)
        print(key, " attack success(far@0.01) rate: ", success001 / total)
        print(key, " attack success(far@0.001) rate: ", success0001 / total)

if __name__ == '__main__':
    main()