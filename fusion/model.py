import torch

from ..accel.models.resnet import ResNetBaseline

class FusionModel(torch.nn.Module):
    def __init__(self, modalities={'audio': 1024, 'accel': 1024, 'video': 1024}, accel_feature_extractor=None):
        """
        """
        super().__init__()

        self.accel_head = accel_feature_extractor
        self.modalities = modalities
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(modalities.get('audio', 0) + modalities.get('accel', 0) + modalities.get('video', 0), 1)
        )

    def forward(self, batch:dict):
        """
        """
        features = []
        if 'accel' in batch:
            features.append(self.accel_head(batch))

        if 'video' in self.modalities:
            features.append(batch['video'])
        
        if 'audio' in self.modalities:
            features.append(batch['audio'])

        return self.linear(torch.hstack(features))
