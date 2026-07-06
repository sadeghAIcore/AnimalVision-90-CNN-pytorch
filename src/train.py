import torch
from torch import nn

from src.engine import train
from src.data_setup import create_dataloader
from src.model import create_effnetB2_model

#import the model and transform function 
effnet_b2_model , effnet_b2_transform= create_effnetB2_model()

train_dir = 'data/animals_link/animals/animals'

device = 'cuda' if torch.cuda.is_available() else 'cpu' #set the device

#set the optimizer
optimizer = torch.optim.Adam(params=effnet_b2_model.parameters(),
                             lr=1e-3)
# Setup loss function
loss_fn = torch.nn.CrossEntropyLoss()

#first we get our train Dataloader and classes
train_Dataloader , classes_name = create_dataloader(transform=effnet_b2_transform , train_dir=train_dir , batch_size=32)

effnetb2_results = train(model=effnet_b2_model,
                                train_dataloader=train_Dataloader,
                                epochs=10,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                device=device)
