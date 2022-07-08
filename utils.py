import torch
import numpy as np
import pandas as pd
import librosa.display

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def load_examples(path):
    df = pd.read_csv(path)
    if 'onset_times' in df.columns:
        df['onset_times'] = [eval(el) for el in df['onset_times']]
    if 'offset_times' in df.columns:
        df['offset_times'] = [eval(el) for el in df['offset_times']]
    return df

def ious(mask, gt_mask):
    mask_area = np.count_nonzero(mask == 1, axis=1)
    gt_mask_area = np.count_nonzero(gt_mask == 1, axis=1)
    intersection = np.count_nonzero( np.logical_and( mask, gt_mask ), axis=1 )
    ious = np.divide(intersection, (mask_area + gt_mask_area - intersection))
    ious[(gt_mask_area == 0) & (mask_area == 0)] = 1
    return ious

def _plot_audio_sample(sample, ax):
    librosa.display.specshow(sample['audio'].squeeze().transpose(), 
                         x_axis='time',
                         y_axis='mel', sr=8000,
                         hop_length=186,
                         fmax=8000, ax=ax)

    m = np.concatenate([[0], sample['label']])
    ini_idx = np.where(np.diff(m) == 1)[0] - 1

    m = np.concatenate([sample['label'], [0]])
    end_idx = np.where(np.diff(m) == -1)[0]
    assert len(ini_idx) == len(end_idx)

    t_max = ax.get_xlim()[1]
    for i, j in zip(ini_idx, end_idx):
        i_t = t_max * i / len(sample['label'])
        j_t = t_max * j / len(sample['label'])
        ax.axvspan(i_t, j_t, color='red', alpha=0.2)

def plot_audio(samples):
    fig, axs = plt.subplots(1,len(samples), figsize=(12,4))
    for ax, sample in zip(axs, samples):
        _plot_audio_sample(sample, ax)

def _plot_accel_sample(sample, ax):
    ax.plot(sample['accel'][0,:])
    ax.plot(sample['accel'][1,:])
    ax.plot(sample['accel'][2,:])

    ax.plot(sample['label'])

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