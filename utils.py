import torch

from sklearn.metrics import roc_auc_score

def get_metrics(outputs, labels, type='binary'):
    if type == 'binary':
        proba = torch.sigmoid(outputs)
        pred = (proba > 0.5)

        correct = pred.eq(outputs.bool()).sum().item()
        return {
            'auc': roc_auc_score(labels, proba),
            'correct': correct
        }
    elif type == 'regression':
        return {
            'mse': torch.nn.functional.mse_loss(outputs, labels, reduction='mean'),
            'l1': torch.nn.functional.l1_loss(outputs, labels, reduction='mean')
        }