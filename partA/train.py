import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import *


BATCH_SIZE = 64
IMAGE_SIZE = (256, 256)
NUM_WORKERS = 4
NUM_CLASSES = 10

def get_transforms(train=False, data_augmentation=False):
    if train and data_augmentation:
        return transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def prepare_datasets(data_augmentation=False):
    train_dataset = datasets.ImageFolder(
        root='./inaturalist_12K/train',
        transform=get_transforms(train=True, data_augmentation=data_augmentation)
    )
    
    val_dataset = datasets.ImageFolder(
        root='./inaturalist_12K/train',
        transform=get_transforms(train=False)
    )
    
    test_dataset = datasets.ImageFolder(
        root='./inaturalist_12K/test',
        transform=get_transforms(train=False)
    )

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, val_indices = next(sss.split(
        range(len(train_dataset)),
        train_dataset.targets
    ))

    return train_dataset, val_dataset, test_dataset, train_indices, val_indices

def create_dataloaders(config):
    train_dataset, val_dataset, test_dataset, train_idx, val_idx = prepare_datasets(
        data_augmentation=config.data_augmentation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(train_idx),
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        sampler=SubsetRandomSampler(val_idx),
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_loader, val_loader, test_loader
    

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        config_dict = dict(config)
        self.save_hyperparameters(config_dict)
        self.model = CustomCNN(
            num_classes=NUM_CLASSES,
            base_filters=config.base_filters,
            filter_organization=config.filter_organization,
            activation=config.activation,
            batch_norm=config.batch_norm,
            dropout_rate=config.dropout_rate,
            dropout_location=config.dropout_location,
            input_size=IMAGE_SIZE
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        optimizers = {
            "adam": torch.optim.Adam,
            "nadam": torch.optim.NAdam,
            "rmsprop": torch.optim.RMSprop
        }
        optimizer = optimizers[self.hparams.optimizer](
            self.parameters(), 
            lr=self.hparams.learning_rate
        )
        return optimizer