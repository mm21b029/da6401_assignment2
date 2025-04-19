import torch
import pytorch_lightning as pl
from torch import nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import WandbLogger

# Configuration
BATCH_SIZE = 64
LR = 1e-3
NUM_CLASSES = 10
PROJECT_NAME = "inaturalist-finetuning"

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Lightning Data Module
class NaturalistDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_dir = "./inaturalist_12K/train"
        self.test_dir = "./inaturalist_12K/test"

    def setup(self, stage=None):
        full_train = ImageFolder(self.train_dir, transform=train_transform)
        self.test_ds = ImageFolder(self.test_dir, transform=val_test_transform)
        
        train_size = int(0.8 * len(full_train))
        val_size = len(full_train) - train_size
        self.train_ds, self.val_ds = torch.utils.data.random_split(
            full_train, [train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=BATCH_SIZE, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=BATCH_SIZE, num_workers=4)


class FineTunedModel(pl.LightningModule):
    def __init__(self, lr=LR):
        super().__init__()
        self.save_hyperparameters()
        
        # Loading and modifying pre-trained model
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, NUM_CLASSES)
        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.hparams.lr)
        return optimizer

    def on_train_start(self):
        self.logger.watch(self.model, log='gradients', log_freq=100)

if __name__ == "__main__":
    dm = NaturalistDataModule()
    model = FineTunedModel()
    
    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        log_model=True,
        save_dir="logs"
    )
    
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_acc',
                mode='max',
                save_top_k=1,
                filename='best-{epoch}-{val_acc:.2f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_acc',
                patience=3,
                mode='max'
            ),
            pl.callbacks.LearningRateMonitor()
        ],
        enable_progress_bar=True
    )
    
    trainer.fit(model, dm)
    trainer.test(model, dm)