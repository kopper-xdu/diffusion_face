class Attack:
    def __init__(self, model) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        