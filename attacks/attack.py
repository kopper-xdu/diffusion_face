import torch

class Attack:
    def __init__(self, model) -> None:
        self.model = model
        self.device = torch.device('cuda')
        