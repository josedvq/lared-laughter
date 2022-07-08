from sklearn.metrics import f1_score
import torch
from pytorch_lightning.utilities.seed import isolate_rng

from ..audio.models.resnet import (
    AudioSegmentationHead,
    get_pretrained_body as get_audio_feature_extractor
)
from ..video.models.models import (
    make_slow_pretrained_body as get_video_feature_extractor, 
    SegmentationHead as VideoSegHead)
from ..accel.models.resnet import (
    ResNetBody as AccelModelBody,
    ResNetBodyNoChannelPool as AccelModelBodyNoPool, 
    SegmentationHead as AccelSeghead
)

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
            self.accel_feature_extractor = AccelModelBody(c_in=3)

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

        if 'video' in batch:
            f = batch['video']
            f = f.reshape(f.shape[0], -1)
            features.append(f)
        
        if 'audio' in batch:
            f = batch['audio']
            f = f.reshape(f.shape[0], -1)
            features.append(f)

        x = self.linear(torch.hstack(features))

        return x
        

class SegmentationFusionModel(torch.nn.Module):
    def __init__(self,
        modalities,
        mask_len=45):
        """
        """
        super().__init__()

        if 'accel' in modalities:
            self.accel_feature_extractor = AccelModelBodyNoPool(c_in=3)

        self.modalities = modalities

        if 'accel' in modalities:
            self.accel_head = AccelSeghead(c_out=1, output_len=mask_len)
        if 'audio' in modalities:
            self.audio_head = AudioSegmentationHead(num_channels=16, output_len=mask_len)
        if 'video' in modalities:
            self.video_head = VideoSegHead(output_len=mask_len)

    def forward(self, batch:dict):
        """
        """
        masks = []
        if 'accel' in batch:
            f = self.accel_feature_extractor(batch['accel'])
            masks.append(self.accel_head(f))

        if 'video' in batch:
            f = batch['video']
            masks.append(self.video_head(f))
        
        if 'audio' in batch:
            f = batch['audio']
            masks.append(self.audio_head(f))

        masks = torch.stack(masks, dim=2)
        masks = masks.mean(dim=2)

        # average over the new mask dim
        return masks