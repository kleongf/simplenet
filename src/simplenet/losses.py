import numpy as np

class Loss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, pred, target):
        raw = self._compute(pred, target)
        if self.reduction == 'mean':
            return raw.mean()
        elif self.reduction == 'sum':
            return raw.sum()
        return raw

    def _compute(self, pred, target):
        raise NotImplementedError

# mean squared error loss
class MSELoss(Loss):
    def _compute(self, pred, target):
        return (pred - target) ** 2

# mean absolute error loss
class MAELoss(Loss):
    def _compute(self, pred, target):
        return (pred - target).abs()

# cross entropy loss: used for classification
class CrossEntropyLoss(Loss):
    def _compute(self, pred, target):
        # pred: (batch, classes) logits, target: (batch,) class indices
        log_probs = pred.softmax(axis=-1).log()
        return -log_probs[np.arange(len(target.data)), target.data]

# binary cross entropy loss: used for binary classification
class BCELoss(Loss):
    def _compute(self, pred, target):
        # pred: (batch, 1) probabilities, target: (batch, 1) binary labels
        return -(target * pred.log() + (1 - target) * (1 - pred).log())
    
