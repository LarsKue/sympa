

class ModelEvaluation:
    def __init__(self, model):
        self.model = model
        self.mode = None

    def __enter__(self):
        self.mode = self.model.training
        self.model.train(mode=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.train(self.mode)
        self.mode = None
