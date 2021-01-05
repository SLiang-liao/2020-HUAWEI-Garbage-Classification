import torch as th
import torchvision.models.densenet as DenseNet



def densenet121(pretrained =True):
    model = DenseNet.densenet121(True)
    model.classifier = th.nn.Sequential(th.nn.Linear(1024,512),th.nn.ReLU(),th.nn.Linear(512,43))

    return model
#print(densenet121())
def densenet169(pretrained =True):
    model = DenseNet.densenet169(True)
    model.classifier = th.nn.Sequential(th.nn.Linear(1024,512),th.nn.ReLU(),th.nn.Linear(512,43))

    return model
#print(densenet169())

def densenet201(pretrained =True):
    model = DenseNet.densenet201(True)
    #model.classifier = th.nn.Sequential(th.nn.Linear(1920,512),th.nn.ReLU(),th.nn.Linear(512,43))

    return model
#print(densenet201())
