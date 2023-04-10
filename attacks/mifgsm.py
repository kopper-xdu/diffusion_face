import torch
import torch.nn as nn
import torch.nn.functional as F


from .attack import Attack


class MIFGSM(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=16/255, alpha=2/255, steps=10, decay=1.0):
        super().__init__(model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        # self.supported_mode = ['default', 'targeted']

    def forward(self, images, tgt_images=None):
        B = images.shape[0]
        images = images.clone().detach()
        
        if tgt_images is not None:
            tgt_images = tgt_images.clone().detach()

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

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

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
