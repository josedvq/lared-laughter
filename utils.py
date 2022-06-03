import torch
import numpy as np
import librosa.display

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def _plot_audio_sample(sample, ax):
    librosa.display.specshow(sample['audio'].squeeze().transpose(), 
                         x_axis='time',
                         y_axis='mel', sr=8000,
                         hop_length=186,
                         fmax=8000, ax=ax)

    m = np.concatenate([[0], sample['seg_mask']])
    ini_idx = np.where(np.diff(m) == 1)[0] - 1

    m = np.concatenate([sample['seg_mask'], [0]])
    end_idx = np.where(np.diff(m) == -1)[0]
    assert len(ini_idx) == len(end_idx)

    t_max = ax.get_xlim()[1]
    for i, j in zip(ini_idx, end_idx):
        i_t = t_max * i / len(sample['seg_mask'])
        j_t = t_max * j / len(sample['seg_mask'])
        ax.axvspan(i_t, j_t, color='red', alpha=0.2)

def plot_audio(samples):
    fig, axs = plt.subplots(1,len(samples), figsize=(12,4))
    for ax, sample in zip(axs, samples):
        _plot_audio_sample(sample, ax)

def _plot_accel_sample(sample, ax):
    ax.plot(sample['accel'][0,:])
    ax.plot(sample['accel'][1,:])
    ax.plot(sample['accel'][2,:])

    ax.plot(sample['seg_mask'])

def plot_accel(samples):
    fig, axs = plt.subplots(1,len(samples), figsize=(12,4))
    for ax, sample in zip(axs, samples):
        _plot_accel_sample(sample, ax)

def get_metrics(outputs, labels, type='classification'):
    if type == 'classification':
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