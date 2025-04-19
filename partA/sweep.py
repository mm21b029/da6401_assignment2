import wandb
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from model import *
import matplotlib.pyplot as plt
import numpy as np
from train import *

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_acc',
        'goal': 'maximize'   
    },
    'parameters': {
        'base_filters': {'values': [16, 32, 64]},
        'filter_organization': {'values': ["same", "double", "half"]},
        'activation': {'values': ["gelu", "silu"]},
        'batch_norm': {'values': [True]},
        'dropout_rate': {'values': [0.0, 0.2, 0.3]},
        'dropout_location': {'values': ["conv", "dense", "both"]},
        'data_augmentation': {'values': [True]},
        'batch_size': {'values': [32, 64]},
        'learning_rate': {
            'min': 1e-4,
            'max': 1e-3,
            'distribution': 'log_uniform_values'
        },
        'optimizer': {'values': ["nadam"]}
    }
}

# with open("../input/sweepfile1/sweep.yaml", "r") as file:
#     sweep_config = yaml.safe_load(file)

def create_prediction_grid(model, test_dir, class_names, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
    
    model.eval()
    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denorm = transforms.Normalize((-mean/std), (1.0/std))
    images = denorm(images).cpu().numpy()
    
    plt.figure(figsize=(15, 50))
    for idx in range(10):
        plt.subplot(10, 3, 3*idx+1)
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        plt.axis('off')
        if idx == 0:
            plt.title("Input Image", fontsize=12, pad=10)
        
        plt.subplot(10, 3, 3*idx+2)
        plt.text(0.5, 0.5, class_names[labels[idx]], 
                ha='center', va='center', fontsize=12)
        plt.axis('off')
        if idx == 0:
            plt.title("True Class", fontsize=12, pad=10)
        
        plt.subplot(10, 3, 3*idx+3)
        color = 'green' if preds[idx] == labels[idx] else 'red'
        plt.text(0.5, 0.5, class_names[preds[idx]], 
                color=color, ha='center', va='center', fontsize=12)
        plt.axis('off')
        if idx == 0:
            plt.title("Predicted Class", fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig('prediction_grid.png', bbox_inches='tight')
    plt.show()

def train():
    with wandb.init() as run:
        config = wandb.config
        run.name = f"bf_{config.base_filters}_fo_{config.filter_organization}_ac_{config.activation}_bn_{config.batch_norm}_dr_{config.dropout_rate}_dl_{config.dropout_location}_da_{config.data_augmentation}_bs_{config.batch_size}_lr_{config.learning_rate}_op_{config.optimizer}"
        
        train_loader, val_loader, test_loader = create_dataloaders(config)

        model = LitModel(config)
        
        trainer = pl.Trainer(
            max_epochs=10,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=pl.loggers.WandbLogger(project="inaturalist"),
            precision=16,
            callbacks=[
                pl.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    mode='min'
                )
            ],
            enable_progress_bar=True,
            enable_model_summary=True
        )
        
        trainer.fit(model, train_loader, val_loader)
        
        trainer.test(model, test_loader)

        test_dir = './inaturalist_12K/test'
        class_names = datasets.ImageFolder(test_dir).classes
        
        create_prediction_grid(model, test_dir, class_names)


sweep_id = wandb.sweep(sweep_config, project="inaturalist")

wandb.agent(sweep_id, function=train, count=50)