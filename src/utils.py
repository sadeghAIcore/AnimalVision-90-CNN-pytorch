
import torch
import os 
import matplotlib.pyplot as plt


def save_model(model):
    if not os.path.exists('../models'):
        os.makedirs('../models')
        print('models folder created')

    torch.save(model.state_dict() , '../models/model.pth')
    print("✅ Model Saved")

def plotresults(train_accs , train_losses):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
