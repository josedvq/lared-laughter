import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

from lared_laughter.accel.models.cnn import MyAlexNet
from lared_laughter.accel.models.resnet import ResNetBaseline

class System(pl.LightningModule):
    def __init__(self, model_name, model_hparams={}, optimizer_name='adam', optimizer_hparams={}):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()

        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()

        self.model = {
            'alexnet': MyAlexNet(),
            'resnet': ResNetBaseline(in_channels=3)
        }[model_name]

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        return self.model(batch['accel'].permute(0, 2, 1).float())

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        X = batch['accel'].permute(0, 2, 1).float()
        Y = batch['label'].float()

        output = self.model(X).squeeze()
        loss = F.binary_cross_entropy_with_logits(output, Y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X = batch['accel'].permute(0, 2, 1).float()
        Y = batch['label'].float()

        output = self.model(X).squeeze()
        val_loss = F.binary_cross_entropy_with_logits(output, Y)
        self.log('val_loss', val_loss)

        return (output, Y.squeeze())

    def validation_epoch_end(self, validation_step_outputs):
        all_outputs = torch.cat([o[0] for o in validation_step_outputs]).cpu()
        all_labels = torch.cat([o[1] for o in validation_step_outputs]).cpu()

        val_auc = roc_auc_score(all_labels, all_outputs)
        self.log('val_auc', val_auc)

    def test_step(self, batch, batch_idx):
        X = batch['accel'].permute(0, 2, 1).float()
        Y = batch['label'].float()

        output = self.model(X).squeeze()

        return (output, Y.squeeze())

    def test_epoch_end(self, test_step_outputs):
        all_outputs = torch.cat([o[0] for o in test_step_outputs]).cpu()
        all_labels = torch.cat([o[1] for o in test_step_outputs]).cpu()

        test_auc = roc_auc_score(all_labels, all_outputs)
        self.test_results = {'auc': test_auc, 'proba': all_outputs}
        self.log('test_auc', test_auc)