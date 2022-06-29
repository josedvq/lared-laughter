import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

from lared_laughter.accel.models.cnn import MyAlexNet
from lared_laughter.accel.models.resnet import ResNet

class System(pl.LightningModule):
    def __init__(self, model_name, task='classification', model_hparams={}, optimizer_name='adam', optimizer_hparams={}):
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
            'resnet': ResNet(c_in=3, c_out=1)
        }[model_name]

        self.loss_fn = {
            'classification':F.binary_cross_entropy_with_logits,
            'regression': F.l1_loss
        }[task]

        self.performance_metric = {
            'classification': lambda input, target: roc_auc_score(target, input),
            'regression': F.l1_loss
        }[task]

    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        return self.model(batch['accel'])

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        X = batch['accel']
        Y = batch['label'].float()

        output = self.model(X).squeeze()
        loss = self.loss_fn(output, Y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        X = batch['accel']
        Y = batch['label'].float()

        output = self.model(X).squeeze()
        val_loss = self.loss_fn(output, Y)
        self.log('val_loss', val_loss)

        return (output, Y.squeeze())

    def validation_epoch_end(self, validation_step_outputs):
        all_outputs = torch.cat([o[0] for o in validation_step_outputs]).cpu()
        all_labels = torch.cat([o[1] for o in validation_step_outputs]).cpu()

        val_metric = self.performance_metric(all_outputs, all_labels)
        self.log('val_metric', val_metric)

    def test_step(self, batch, batch_idx):
        X = batch['accel']
        Y = batch['label'].float()

        output = self.model(X).squeeze()

        return (output, Y.squeeze())

    def test_epoch_end(self, test_step_outputs):
        all_outputs = torch.cat([o[0] for o in test_step_outputs]).cpu()
        all_labels = torch.cat([o[1] for o in test_step_outputs]).cpu()

        test_metric = self.performance_metric(all_outputs, all_labels)
        self.test_results = {'metric': test_metric, 'proba': all_outputs}
        self.log('test_metric', test_metric)

