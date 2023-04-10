import torch
from torch import nn
import torch.nn.functional as F

from .attack import Attack


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=16/255):
        super().__init__(model)
        self.eps = eps
        
        # self.attack = 'FGSM'


    def forward(self, images, tgt_images=None):
        B = images.shape[0]
        
        images = images.clone().detach()
        images.requires_grad = True
        
        if tgt_images is not None:
            tgt_images = tgt_images.clone().detach()
        
        loss = 0
        for name, model in self.model.items():
            model.eval()
            resize = (112, 112) if name != 'FaceNet' else None
            
            if resize is not None:
                images = F.interpolate(images, resize, mode='bilinear', align_corners=True)
                tgt_images = F.interpolate(tgt_images, resize, mode='bilinear', align_corners=True)

            fea1 = model(images * 2 - 1).reshape(B, -1)
            
            if tgt_images is not None:
                fea2 = model(tgt_images * 2 - 1).reshape(B, -1)
                loss += F.cosine_similarity(fea1, fea2).sum() / B
            else:
                loss += -F.cosine_similarity(fea1, fea1.clone().detach()).sum() / B

        # Update adversarial images
        grad = torch.autograd.grad(loss / len(self.model), images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images
