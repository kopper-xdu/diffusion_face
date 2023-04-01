import torch
from torch import nn
import numpy as np
import random
from omegaconf import OmegaConf
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

import os
from fr_model import IRSE_50, MobileFaceNet, IR_152, InceptionResnetV1
from FaceParsing.interface import FaceParsing
from dataset import base_dataset
from torchvision import transforms
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import Subset

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


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)
    for param in model.parameters():
        param.requires_grad = False

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    seed = 0
    h = 512
    w = 512
    txt = ''
    ddim_steps = 45
    scale = 0
    classifier_scale = 300
    batch_size = 2
    num_workers = 0
    
    setup_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    transform = transforms.Compose([transforms.Resize((512, 512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = base_dataset(path='./data', transform=transform)
    dataset = Subset(dataset, [x for x in range(50)])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # x_target = Image.open('17082.png').convert('RGB')
    # trans = transforms.Compose([transforms.Resize((112, 112)),
    #                             transforms.ToTensor(),
    #                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # x_target = trans(x_target)
    
    sampler = initialize_model('configs/stable-diffusion/v2-inpainting-inference.yaml', 
                               'pretrained_model/512-inpainting-ema.ckpt')
    model = sampler.model
    # classifier = get_model('IRSE50')

    prng = np.random.RandomState(seed)
    start_code = prng.randn(batch_size, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    # with torch.no_grad(), \
    #         torch.autocast("cuda"):
    # attack_model_names = ['IR152', 'IRSE50', 'FaceNet', 'MobileFace']
    attack_model_names = ['IR152']
    attack_model_dict = {'IR152': get_model('IR152'), 'IRSE50': get_model('IRSE50'), 
                         'FaceNet': get_model('FaceNet'), 'MobileFace': get_model('MobileFace')}
    # attack_model_resize_dict = {'IR152': 112, 'IRSE50': 112, 'FaceNet': 160, 'MobileFace': 112}
    # cos_sim_scores_dict = {'IR152': [], 'IRSE50': [], 'FaceNet': [], 'MobileFace': []}
    cos_sim_scores_dict = {'IR152': []}
    
    for attack_model_name in attack_model_names:
        attack_model = attack_model_dict[attack_model_name]
        classifier = {k: v for k, v in attack_model_dict.items() if k != attack_model_name}
        resize = nn.AdaptiveAvgPool2d((112, 112)) if attack_model_name != 'FaceNet' else nn.AdaptiveAvgPool2d((160, 160))
        with torch.no_grad():
            for i, (image, tgt_image) in enumerate(dataloader):
                tgt_image = tgt_image.to(device)
                B = image.shape[0]
                
                face_parsing = FaceParsing()
                pred = face_parsing(image)

# label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 
# 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
                def get_mask(number):
                    return pred == number
                
                masks = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]
                # skin = get_mask(1)
                # nose = get_mask(2)
                # eye_gla = get_mask(3)
                # l_eye = get_mask(4)
                # r_eye = get_mask(5)
                # l_brow = get_mask(6)
                # r_brow = get_mask(7)
                # mouth = get_mask(10)
                # u_lip = get_mask(11)
                # l_lip = get_mask(12)
                mask = None
                for x in masks:
                    if mask is not None:
                        mask |= get_mask(x)
                    else:
                        mask = get_mask(x)
                mask = (mask == 0).float().reshape(B, 1, h, w)

                masked_image = image * (mask < 0.5)

                batch = {
                    "image": image.to(device),
                    "txt": batch_size * [txt],
                    "mask": mask.to(device),
                    "masked_image": masked_image.to(device),
                }

                c = model.cond_stage_model.encode(batch["txt"])
                c_cat = list()
                for ck in model.concat_keys:
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        bchw = [batch_size, 4, h // 8, w // 8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond = {"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(batch_size, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, h // 8, w // 8]
                
                samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    batch_size,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
                    log_every_t=10,
                    classifier=classifier,
                    classifier_scale=classifier_scale,
                    x_target=tgt_image
                )
                
                # model.first_stage_model.post_quant_conv.to(0)
                # model.first_stage_model.decoder.to(0)
                x_samples_ddim = model.decode_first_stage(samples_cfg)
                result = torch.clamp(x_samples_ddim, min=-1, max=1)
                # x_inter = torch.clamp(model.decode_first_stage(intermediates['x_inter'][-2]), min=-1, max=1)

                save_image((result + 1) / 2, f'res/{i}.png')
                save_image((masked_image + 1) / 2, f'res/{i}_m.png')

                # save_image((x_inter + 1) / 2, f'res/{i}_inter.png')
                
                # attack_model = attack_model_dict[attack_model_name]
                feature1 = attack_model(resize(result)).reshape(B, -1)
                feature2 = attack_model(resize(tgt_image)).reshape(B, -1)
                
                score = F.cosine_similarity(feature1, feature2)
                print(score)
                cos_sim_scores_dict[attack_model_name] += score.tolist()

                
                # feature3 = attack_model(resize(x_inter)).reshape(B, -1)
                # score = F.cosine_similarity(feature3, feature2)
                # print(score)

    
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
    
main()