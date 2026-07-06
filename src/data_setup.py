import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

num_of_workers = os.cpu_count()

# we use this function to make our test and train Dataloaders

def create_dataloader(transform : int , 
                      batch_size: int , 
                      train_dir:str ,
                      num_of_workers = num_of_workers):

  train_data =  datasets.ImageFolder(train_dir,transform=transform)

  classes_name = train_data.classes 

  train_Dataloder = DataLoader(
      train_data ,
      batch_size = batch_size , 
      shuffle = True , 
      num_workers = num_of_workers
  )

  return train_Dataloder , classes_name


