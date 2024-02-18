import warnings
warnings.filterwarnings('ignore')
import torch
import torch. multiprocessing as mp
import torch._dynamo
from shutil import rmtree
from modules import model, data, train
from pathlib import Path

mp.set_start_method('spawn')
torch._dynamo.config.suppress_errors = True


def train_and_save(data_path:str=str("./data"),
                   batch_size:int=128,
                   label_smoothing:float=0.1,
                   lr:float=1e-3,
                   epochs:int=30,
                   compile:bool=True
                   ):
  """
  Trains and saves the model. Returns the train loss, train accuracy, test loss,
  and test accuracy.

  Keyword Arguments:
    :arg data_path: Path to data to save. Default: "./data"
    :type data_path: string
    :arg batch_size: batch size for chunking the data to smaller pieces. 
     Default 128.
    :type batch_size: int 
    :arg label_smoothing: Label smoothing to make model to thing about its 
     prediction. Default 0.1
    :type label_smoothing: float
    :arg lr: Learning rate. Default 1e-3
    :type lr: float
    :arg epochs: How many epochs to use. Default 30.
    :type epochs: int 
    :arg compile: Use torch2.x's compile method. Default True.
    :type compile: bool

  Example Usage:
    results = train_and_save(data_path="./data",
                             batch_size=128,
                             label_smoothing=0.1,
                             lr=1e-3,
                             epochs=30,
                             compile=True)
  """


  device = "cuda" if torch.cuda.is_available() else "cpu"

  data_path = Path(data_path)
  if data_path.is_dir():
    rmtree(data_path)
  data_path.mkdir(parents=True, exist_ok=True)

  effnetb2_v2_m, train_transforms, test_transforms = model.create_effnetb2_v2_m(102)
  train_dataloader, test_dataloader, val_dataloader = data.create_dataloaders(root=data_path,
                                                                              train_transforms=train_transforms,
                                                                              test_transforms=test_transforms,
                                                                              batch_size=batch_size,
                                                                              device="cpu")

  effnetb2_v2_m.to(device)

  loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
  optimizer = torch.optim.Adam(effnetb2_v2_m.parameters(), lr=lr)

  if compile:
    compile_model = torch.compile(effnetb2_v2_m)

  results = train.train_model(model=compile_model,
                              train_dataloader=train_dataloader,
                              test_dataloader=test_dataloader,
                              optimizer=optimizer,
                              loss_fn=loss_fn,
                              device=device,
                              epochs=epochs)

  torch.save(obj=effnetb2_v2_m.state_dict(),f="./flower102_effnetb2_v2_m.pth")

  return results
