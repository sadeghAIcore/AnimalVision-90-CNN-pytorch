# the we should put model.py , model.pth in this folder
# then we starting to duild our app.py file to push it on hugging face

import torch
import os
from timeit import default_timer as timer
from model import create_effnetB2_model
from pathlib import Path

model , transform = create_effnetB2_model()
model.load_state_dict(torch.load('model.pth' , map_location='cpu'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle',
               'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee',
               'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin', 
               'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox',
               'goat', 'goldfish', 'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog',
               'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala',
               'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan',
               'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum',
               'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel',
               'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra']

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

title = "AnimalVision-90 🐈🐶🐄"
description = 'A deep learning model for identifying 90 different animal species from images with high accuracy.'

example_list = list(Path('example').glob('*.jpg'))

demo = gr.Interface(
    fn = predict,
    inputs = gr.Image(type="pil") ,
    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
            gr.Number(label="Prediction time (s)")] ,
    examples=example_list, 
    title=title,
    description=description
)

demo.launch(debug=False, # print errors locally?
            share=True)    
