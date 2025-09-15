"""
this is rgbn sigle feature extrator code with both slope and without slope
"""
import torchvision.models as models
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import trainer
from pytorch_lightning.loggers import TensorBoardLogger,CSVLogger
from glob import glob
import cv2
import os
import numpy as np
from PIL import Image
from model.RGB import RGBStreamSwin
from model.RGBNNet import RGBNStreamSwin
import sys
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import time
from config import load_config


class CustomDataset(Dataset):
    def __init__(self, data_path, Train,name, transform=None ):
        self.data_path = data_path
        self.img_list_r = glob(data_path['r'])
        self.img_list_g = glob(data_path['g'])
        self.img_list_b = glob(data_path['b'])
        self.img_list_n = glob(data_path['n'])
        self.flag = Train
        self.transform = transform
        self.name= name
        self.material_mapping = {
            'vinyl': 0,
            'fabric': 1,
            'paper': 2,
            'ceremic': 3,
            'metal': 4,
            'stone': 5,
        }
        
    def __len__(self):
        if len(self.img_list_r) == len(self.img_list_g) == len(self.img_list_b) == len(self.img_list_n):
            return len(self.img_list_r)
        else:
            print("Length of the materials are not same.")  
            return 0  # to avoid index errors

    def __getitem__(self, idx):
        img_r = cv2.imread(self.img_list_r[idx], cv2.IMREAD_UNCHANGED)
        if img_r is None:
            raise RuntimeError(f"Could not load image: {self.img_list_r[idx]}")
            sys.exit()
        img_g = cv2.imread(self.img_list_g[idx], cv2.IMREAD_UNCHANGED)
        img_b = cv2.imread(self.img_list_b[idx], cv2.IMREAD_UNCHANGED)
        img_n = cv2.imread(self.img_list_n[idx], cv2.IMREAD_UNCHANGED)

        img_r = torch.tensor(img_r).float()
        img_g = torch.tensor(img_g).float()
        img_b = torch.tensor(img_b).float()
        img_n = torch.tensor(img_n).float()

        material_obj_type_n = "_".join(os.path.basename(self.img_list_n[idx]).split('_')[0:3])
        material_obj_type_b = "_".join(os.path.basename(self.img_list_b[idx]).split('_')[0:3])
        material_obj_type_g = "_".join(os.path.basename(self.img_list_g[idx]).split('_')[0:3])
        material_obj_type_r = "_".join(os.path.basename(self.img_list_r[idx]).split('_')[0:3])
        if not material_obj_type_r == material_obj_type_g == material_obj_type_b == material_obj_type_n:
            print("Error: The material types of R, G, B, and N do not match.")
            sys.exit("Error: The material types of R, G, B, and N do not match.")

        img_r=img_r/255
        img_g=img_g/255
        img_b=img_b/255
        img_n=img_n/255


        # Extract material and object types from the filenames
        material_type_r = os.path.basename(self.img_list_r[idx]).split('_')[1]
        obj_type_r = "_".join(os.path.basename(self.img_list_r[idx]).split('_')[2])
        
        material_type_g = os.path.basename(self.img_list_g[idx]).split('_')[1]
        obj_type_g = "_".join(os.path.basename(self.img_list_g[idx]).split('_')[2])
        
        material_type_b = os.path.basename(self.img_list_b[idx]).split('_')[1]
        obj_type_b = "_".join(os.path.basename(self.img_list_b[idx]).split('_')[2])
        
        material_type_n = os.path.basename(self.img_list_n[idx]).split('_')[1]
        obj_type_n = "_".join(os.path.basename(self.img_list_n[idx]).split('_')[2])
        


        if material_type_r == material_type_g == material_type_b == material_type_n:
            if material_type_n not in self.material_mapping:
                print("the material class is not present ")
                sys.exit()
            label = self.material_mapping.get(material_type_n) 
            # Ensure all channels match
            if obj_type_r == obj_type_g == obj_type_b == obj_type_n:
                cat_data = torch.stack([img_r, img_g, img_b], dim=0)  # Concatenate along C dimension
                # s_cat_data = torch.stack([s_img_r, s_img_g, s_img_b], dim=0)  # Concatenate along C dimension
            else:
                print("Error: The material types of R, G, B, and N do not match.")
                sys.exit("Error: the obj types of R, G, B, and N do not match.")
        else:
            print("Error: The material types of R, G, B, and N do not match.")
        img_n = img_n.unsqueeze(0)
        sample = {'image': cat_data,"img_n":img_n, 'labels': label}
        return sample

    
# Neural Network using PyTorch Lightning
class NeuralNet(pl.LightningModule):
    def __init__(self,  output_classes,name):
        super(NeuralNet, self).__init__()
        self.name=name
        if self.name=="rgbn":
            self.model = RGBNStreamSwin(output_classes)
        elif self.name=="rgb":
            self.model= RGBStreamSwin(output_classes)
        

    def forward(self, rgb,nir):
        if self.name=="rgbn":
            return self.model(rgb, nir)
        elif self.name=="rgb":
            return self.model(rgb)

    def training_step(self, batch, batch_idx):

        rgb_images = batch['image']
        nir_images = batch["img_n"]
        labels = batch['labels']

        out = self.forward(rgb_images,nir_images)
        loss = F.cross_entropy(out, labels)
        self.log("train_loss", loss, prog_bar=True,on_step=False, on_epoch=True, logger=True)

        
        preds = torch.argmax(out, dim=1)
        correct = (preds == labels).float().sum()
        accuracy = correct / len(labels)
        self.log("train_acc", accuracy,on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss,"train_acc":accuracy}

    def validation_step(self, batch, batch_idx):
  
        rgb_images = batch['image']
        nir_images = batch["img_n"]
        labels = batch['labels']

        # images = images.float()
        out = self.forward(rgb_images,nir_images)
        loss = F.cross_entropy(out, labels)
        self.log("val_loss", loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

         
        # Calculate accuracy
        preds = torch.argmax(out, dim=1)
        correct = (preds == labels).float().sum()
        accuracy = correct / len(labels)
        self.log("val_accuracy", accuracy,on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"val_loss": loss ,"val_accuracy": accuracy}
    
    def test_step(self, batch, batch_idx):
  
        rgb_images = batch['image']
        nir_images = batch["img_n"]
        labels = batch['labels']

        # images = images.float()
        out = self.forward(rgb_images,nir_images)

        # loss
        loss = F.cross_entropy(out, labels)
        self.log("t_val_loss", loss,on_step=False, on_epoch=True, prog_bar=True, logger=True)

                
        # Calculate accuracy
        preds = torch.argmax(out, dim=1)
        correct = (preds == labels).float().sum()
        accuracy = correct / len(labels)
        self.log("t_val_accuracy", accuracy,on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"t_val_loss": loss ,"t_val_accuracy": accuracy}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=config["training"]["lr"])
        return optimizer

    def train_dataloader(self):
        transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),  # or random.choice([0, 90, 180, 270])
        ])

        # /astori/data3/refined
        paths = {
            'r': r"/astori/data3/refined/_r512/*.png",
            'g': r"/astori/data3/refined/_g512/*.png",
            'b': r"/astori/data3/refined/_b512/*.png",
            'n': r"/astori/data3/refined/_nir512/*.png"
        }
        paths = config["data"]["train"]
        train_dataset = CustomDataset(data_path=paths, Train=True,name=self.name, transform=None)
        print(f"Training dataset size: {len(train_dataset)}")
        return DataLoader(dataset=train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)

    def val_dataloader(self):

        paths = config["data"]["val"]
        print(paths)
        val_dataset = CustomDataset(data_path=paths, Train=False, name=self.name, transform=None)
        print(f"Validation dataset size: {len(val_dataset)}")  # Ensure data is loaded correctly
        return DataLoader(dataset=val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    def test_dataloader(self):
        paths = config["data"]["test"]
        test_dataset = CustomDataset(data_path=paths, Train=False, name=self.name, transform=None)
        print(f"Test dataset size: {len(test_dataset)}")
        return DataLoader(dataset=test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    

early_stop_callback = EarlyStopping(
    monitor="val_accuracy",    # Monitor validation accuracy
    patience=5,           # Stop training if val_acc doesn't improve for 5 consecutive epochs
    mode="max",           # Higher val_acc is better
    verbose=True
)

best_model_checkpoint = ModelCheckpoint(
    monitor="val_accuracy",          # Monitor validation accuracy
    mode="max",                 # Save model with highest val_acc
    save_top_k=14,               # Only keep the best model
    # filename="b_{epoch:02d}-{val_acc:.4f}",  # Naming format
    filename="b_epoch{epoch:02d}",  # Naming format
    save_weights_only=False,
    verbose=True
)

last_n_checkpoints = ModelCheckpoint(
    monitor="epoch",
    mode="max",               # higher epoch = later
    save_top_k=10,            # keep only the last 10 checkpoints
    filename="last_epoch{epoch:02d}",
    save_weights_only=False,
    verbose=True
)
# Always save the last model at the end of training
last_model_checkpoint = ModelCheckpoint(
    save_last=True,             # Save last model (automatically named "last.ckpt")
    verbose=True
)
periodic_checkpoint = ModelCheckpoint(
    every_n_epochs=5,      
    save_top_k=-1,          # -1 = save all
    filename="p_epoch-{epoch:02d}",
    verbose=True
)



if __name__ == "__main__":
    # 🔹 Load YAML config
    config = load_config("config.yaml")
    version = config["experiment"]["version"]
    settings = config["experiment"]["settings"]
    log_dir = config["experiment"]["log_dir"]
    checkpoints_dir = config["checkpoints_dir"]
    for name in settings:
        logger = TensorBoardLogger(log_dir, name=name, version=version)
        csv_logger = CSVLogger(log_dir, name=f"{name}_csv", version=version, flush_logs_every_n_steps=1)

        trainer = pl.Trainer(
            callbacks=[last_n_checkpoints],  # <-- your callback
            logger=[csv_logger, logger],
            max_epochs=config["training"]["num_epochs"],
            accelerator="gpu",
            devices=1,
            fast_dev_run=False,
            accumulate_grad_batches=config["training"]["accumulate_grad_batches"],
        )

        # Model
        model = NeuralNet(config["training"]["num_classes"], name)

        # Training
        start_time = time.time()
        trainer.fit(model)

        # Test checkpoints from last 10 epochs
        last_epochs = range(config["training"]["num_epochs"] - 10, config["training"]["num_epochs"])
        for epoch in last_epochs:
            checkpoint_path = f"{checkpoints_dir}/{name}_csv/{version}/checkpoints/last_epochepoch={epoch}.ckpt"
            print(f"Testing checkpoint: {checkpoint_path}")
            trainer.test(model, ckpt_path=checkpoint_path)

        # Timing stats
        end_time = time.time()
        total_seconds = end_time - start_time
        time_per_epoch_minutes = (total_seconds / config["training"]["num_epochs"]) / 60
        print(f"\n Estimated Time Per Epoch for '{name}': {time_per_epoch_minutes:.2f} minutes\n")

        # Dummy sample input for TensorBoard graph
        rgb_sample_input = torch.rand(1, 3, 224, 224)
        nir_sample_input = torch.rand(1, 1, 224, 224)
        sample_input = (rgb_sample_input, nir_sample_input)

        try:
            logger.experiment.add_graph(model, sample_input)
        except Exception as e:
            print(f" Error adding graph to TensorBoard: {e}")
