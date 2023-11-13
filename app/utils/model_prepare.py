import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms as T
import torch.nn as nn

def class_label(pred):
    dict = {0:'cat', 1 : 'dog'}
    prediction = dict[pred]
    return prediction

def load_model():
    model = resnet18()
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512,1)
    model.load_state_dict(torch.load('app/utils/savemodel.pt'), strict=False)
    model.eval()
    return model



def transform_image(img):

    trnsfrms = T.Compose(
        [
            T.Resize((224,224)),
            T.ToTensor()

        ]


    )
    return trnsfrms(img)
