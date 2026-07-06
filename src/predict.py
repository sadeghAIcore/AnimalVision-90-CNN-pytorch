
import torchvision
import torch
from timeit import default_timer as timer
from src.model import create_effnetB2_model
from src.data_setup import create_dataloader
# we set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_dir = 'data/animals/test'
test_data , class_name = create_dataloader(train_dir=test_dir , batch_size=32 , transform=transform)

# import the model and model transform
model , transform = create_effnetB2_model()
model.load_state_dict(torch.load('models/model.pth'))


def predict(img , topk = 3):

    # starting the timer
    start_time = timer()

    # unsqueeze the image to fit the model 
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    model.to(device)
    with torch.inference_mode():

        pred_probs = torch.softmax(model(img) , dim=1)

        # We use topk to have 3 with the most probability
        top_probs, top_idxs = torch.topk(pred_probs, k=topk, dim=1)

    # make a dict of label probability
    pred_labels_and_probs = {
        class_names[idx]: float(prob)
        for idx, prob in zip(top_idxs[0], top_probs[0])
    }


    pred_time = round(timer() - start_time , 5)

    return pred_labels_and_probs , pred_time
