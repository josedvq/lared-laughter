import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

from lared_laughter.accel.models.cnn import MyAlexNet
from lared_laughter.accel.models.resnet import ResNet, SegmentationResnet

class SegmentationSystem(pl.LightningModule):
    def __init__(self, model_hparams={}, optimizer_name='adam', optimizer_hparams={}):
        """
        Inputs:
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.model = SegmentationResnet(3, 1)

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        return self.model(batch['accel'].permute(0, 2, 1).float())

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        X = batch['accel'].permute(0, 2, 1).float()

        out_proba, out_segmentation = self.model(X)
        out_segmentation = out_segmentation.squeeze()
        loss = F.binary_cross_entropy_with_logits(out_segmentation, batch['seg_mask'])

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X = batch['accel'].permute(0, 2, 1).float()

        out_proba, out_segmentation = self.model(X)
        out_segmentation = out_segmentation.squeeze()
        val_loss = F.binary_cross_entropy_with_logits(out_segmentation, batch['seg_mask'])
        self.log('val_loss', val_loss)

        return (out_segmentation, batch['seg_mask'])

    def validation_epoch_end(self, validation_step_outputs):
        all_outputs = torch.cat([o[0] for o in validation_step_outputs]).cpu()
        all_masks = torch.cat([o[1] for o in validation_step_outputs]).cpu()

        output_masks = (torch.nn.functional.sigmoid(all_outputs) > 0.5).int()

        val_acc = torch.sum(output_masks == all_masks) / all_masks.numel()
        val_loss = F.binary_cross_entropy_with_logits(all_outputs, all_masks)
        self.log('val_acc', val_acc)
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        X = batch['accel'].permute(0, 2, 1).float()

        out_proba, out_segmentation = self.model(X)
        out_segmentation = out_segmentation.squeeze()

        return (out_segmentation, batch['seg_mask'])

    def test_epoch_end(self, test_step_outputs):
        all_outputs = torch.cat([o[0] for o in test_step_outputs]).cpu()
        all_masks = torch.cat([o[1] for o in test_step_outputs]).cpu()

        output_masks = (torch.nn.functional.sigmoid(all_outputs) > 0.5).int()

        test_acc = torch.sum(output_masks == all_masks) / all_masks.numel()
        test_loss = F.binary_cross_entropy_with_logits(all_outputs, all_masks)
        self.test_results = {'acc': test_acc, 'loss': test_loss}
        self.log('test_acc', test_acc)
        self.log('test_loss', test_loss)