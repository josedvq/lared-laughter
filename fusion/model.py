from sklearn.metrics import f1_score
import torch
from ..audio.models.resnet import get_pretrained_body as get_audio_feature_extractor
from ..video.models.models import make_slow_pretrained_body as get_video_feature_extractor
from ..accel.models.resnet import ResNetBody

class FusionModel(torch.nn.Module):
    def __init__(self, 
        modalities):
        """
        """
        super().__init__()

        feature_sizes = {
            'audio': 2304,
            'video': 8192,
            'accel': 128
        }

        if 'accel' in modalities:
            self.accel_feature_extractor = ResNetBody(c_in=3)
        if 'audio' in modalities:
            self.audio_feature_extractor = get_audio_feature_extractor()
        if 'video' in modalities:
            self.video_feature_extractor = get_video_feature_extractor()

        self.modalities = modalities
        num_features = sum([feature_sizes[m] for m in modalities])
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(num_features, 1)
        )

    def forward(self, batch:dict):
        """
        """
        features = []
        if 'accel' in batch:
            features.append(self.accel_feature_extractor(batch['accel']))

        if 'video' in self.modalities:
            f = self.video_feature_extractor(batch['video'])
            f = f.reshape(f.shape[0], -1)
            features.append(f)
        
        if 'audio' in self.modalities:
            f = self.audio_feature_extractor(batch['audio'])
            f = f.reshape(f.shape[0], -1)
            features.append(f)

        return self.linear(torch.hstack(features))
        