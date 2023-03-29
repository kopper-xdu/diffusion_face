import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

import os
from classifier import IRSE_50
from FaceParsing.interface import FaceParsing

torch.set_grad_enabled(False)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3"


def get_classifier():
    classifier = IRSE_50()
    classifier.load_state_dict(torch.load('irse50.pth', map_location=torch.device('cpu')))
    classifier.eval()
    
    return classifier


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


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    # mask = np.array(mask.convert("L"))
    # mask = mask.astype(np.float32) / 255.0
    # mask = mask[None, None]
    # mask[mask < 0.5] = 0
    # mask[mask >= 0.5] = 1
    # mask = torch.from_numpy(mask)
    
    face_parsing = FaceParsing()
    pred = face_parsing(image)
    mask_hair = (pred == 13).reshape(1, 1, 512, 512)
    mask_bg = (pred == 0).reshape(1, 1, 512, 512)
    mask = (mask_bg | mask_hair).float()

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, classifier_scale=10.0, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    # with torch.no_grad(), \
    #         torch.autocast("cuda"):
    with torch.no_grad():
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        
        # classifier guidance
        from PIL import Image
        from torchvision import transforms
        x_target = Image.open('17082.png').convert('RGB')
        trans = transforms.Compose([transforms.Resize((112, 112)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        x_target = trans(x_target)
        
        # guidance_dict = {}
        # guidance_dict['classifier'] = get_classifier().to(device)
        # guidance_dict['classifer_scale'] = classifer_scale
        # guidance_dict['x_target'] = x_target
        
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
            classifier=get_classifier(),
            classifier_scale = classifier_scale,
            x_target = x_target
        )
        model.to(0)
        x_samples_ddim = model.decode_first_stage(samples_cfg)
        
        import torch.nn.functional as F
        B = x_samples_ddim.shape[0]
        classifier = get_classifier()
        reshape = torch.nn.AdaptiveAvgPool2d((112, 112))
        feature1 = classifier(reshape(x_samples_ddim.cpu())).reshape(B, -1)
        feature2 = classifier(reshape(x_target.unsqueeze(0).cpu())).reshape(1, -1)
        feature2 = feature2.repeat(B, 1)
        score = (1 + F.cosine_similarity(feature1, feature2)) / 2
        print(score)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [Image.fromarray(img.astype(np.uint8)) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

def predict(input_image, prompt, ddim_steps, num_samples, scale, seed, classifer_scale):
    init_image = input_image["image"].convert("RGB")
    init_mask = input_image["mask"].convert("RGB")
    
    # init_mask.save('mask.png')
    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        classifier_scale=classifer_scale,
        num_samples=num_samples,
        h=height, w=width
    )

    return result


sampler = initialize_model(sys.argv[1], sys.argv[2])

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Diffusion Inpainting")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=4, value=2, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=500, value=45, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=30, value=10, step=0.1
                )
                classifer_scale = gr.Slider(
                    label="Classifer Guidance Scale", minimum=0, maximum=1000, value=10, step=1
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto")

    run_button.click(fn=predict, inputs=[
                     input_image, prompt, ddim_steps, num_samples, scale, seed, classifer_scale], outputs=[gallery])


block.launch()


