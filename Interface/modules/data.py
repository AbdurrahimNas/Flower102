
import torch 
import torchvision
import os
from torch.utils.data import DataLoader

def create_dataloaders(root:str,
                       train_transforms:torchvision.transforms,
                       test_transforms:torchvision.transforms,
                       batch_size:int=32,
                       num_workers:int=os.cpu_count(),
                       device:torch.device="cuda"
                       ):
  """
  Creates train, test, and validation dataloaders.

  Keyword Arguments:
    :arg root: path of the data
    :type root: str 
    :arg train_transforms: train_transforms
    :type train_transforms: torchvision.transforms
    :arg test_transforms: test_transforms
    :type: test_transforms: torchvision.train_transforms
    :arg batch_size: batch_size
    :type batch_size:int 
    :arg num_workers: num_workers
    :type num_workers: int 
    :arg device: device
    :type device: torch.device

  Example Usage:
    train_dataloader, test_dataloader, val_dataloader = create_dataloaders(root=data_path,
                                                                          train_transforms=train_transforms,
                                                                          test_transforms=test_transforms,
                                                                          batch_size=32,
                                                                          num_workers=os.cpu_count(),
                                                                          device="cuda")
  """
  train_dataset = torchvision.datasets.Flowers102(root=root,
                                                  split="test",
                                                  download=True,
                                                  transform=train_transforms)

  val_dataset = torchvision.datasets.Flowers102(root=root,
                                                split="val",
                                                download=True,
                                                transform=test_transforms)

  test_dataset = torchvision.datasets.Flowers102(root=root,
                                                split="train",
                                                download=True,
                                                transform=test_transforms)

  train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size = batch_size,
                                shuffle=True,
                                num_workers=num_workers,
                                generator=torch.Generator(device=device),
                                pin_memory=True)

  test_dataloader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              generator=torch.Generator(device=device),
                              pin_memory=True)

  val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              generator=torch.Generator(device=device),
                              pin_memory=True)

  print(f"Length Train DataLoader: {len(train_dataloader)} | Length Test DataLoader: {len(test_dataloader)} | Length Val DataLoader: {len(val_dataloader)}")

  return train_dataloader, test_dataloader, val_dataloader
