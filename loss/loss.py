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
    
class MSELoss(Loss):
    def _compute(self, pred, target):
        return (pred - target) ** 2

class CrossEntropyLoss(Loss):
    def _compute(self, pred, target):
        # pred: (batch, classes) logits, target: (batch,) class indices
        log_probs = pred.log_softmax(axis=-1)
        return -log_probs[np.arange(len(target.data)), target.data]