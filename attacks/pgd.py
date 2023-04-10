import torch
import torch.nn as nn
import torch.nn.functional as F


from .attack import Attack


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """
    def __init__(self, model, eps=16/255,
                 alpha=2/255, steps=10, random_start=True):
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        # self.supported_mode = ['default', 'targeted']

    def forward(self, images, tgt_images=None):
        B = images.shape[0]
        images = images.clone().detach()

        if tgt_images is not None:
            tgt_images = tgt_images.clone().detach()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images += torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        
        for _ in range(self.steps):
            adv_images.requires_grad = True
            
            cost = 0
            for name, model in self.model.items():
                model.eval()
                resize = nn.AdaptiveAvgPool2d((112, 112)) if name != 'FaceNet' else nn.AdaptiveAvgPool2d((160, 160))
                
                fea1 = model(resize(adv_images * 2 - 1))
        
                if tgt_images is not None:
                    fea2 = model(resize(tgt_images * 2 - 1))
                    cost += F.cosine_similarity(fea1, fea2).sum() / B
                else:
                    cost += -F.cosine_similarity(fea1, fea1.clone().detach()).sum() / B

                # Update adversarial images
            grad = torch.autograd.grad(cost / len(self.model), adv_images,
                                        retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
