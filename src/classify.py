import utils
import torch.nn as nn
from utils import utils as ut
import torch
from training import training_model as tm
from model import TL
from model import AlexNet as malex
import pandas as pd
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import optim
from functools import partial
from ray import tune
#from ray.air import Checkpoint, Session
from ray.tune.schedulers import ASHAScheduler

import yaml

pathData = "./data"
pathImage = "./images/"
path2Weights = "./weights/resnet_18.pt"
gridSize = 4
num_epochs = 30
sanity_check = False

nameTrain = 'sampleImagesTrain.png'
nameVal = 'sampleImagesVal.png'
nameTest = 'sampleImagesTest.png'

lossGraph = "loss.png"
metricGraph = "metric.png"

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

num_classes = 10
data = ut.STL10Dataset(pathData)
# data.getRandomImages(gridSize,pathImage,nameTrain,'train')
# data.getRandomImages(gridSize,pathImage,nameVal,'val')
# data.getRandomImages(gridSize,pathImage,nameTest,'test')

model= TL.TLModel(num_classes)
model_Alex = malex.AlexNet(num_epochs)
training = tm.Training(model.model_resnet18,num_epochs,
                       data.train_dl,data.val_dl,
                       sanity_check,10,
                       path2Weights)
loss_history , metric_history =training.training_loop()
# ut.plotGraphs(loss_history,"loss",num_epochs,pathImage,lossGraph)
# ut.plotGraphs(metric_history,"metric",num_epochs,pathImage,metricGraph)
def settingRayTune(num_epochs):
        config = {
            "lr": tune.loguniform(1e-4,1e-1),
            "batch_size" : tune.choice([2,4,8,16]),
        }

        scheduler = ASHAScheduler(metric="loss",mode="min",
                                  max_t=num_epochs,grace_period=1,
                                  reduction_factor=2,)

        return config , scheduler

# def main():
#     config , scheduler = settingRayTune(10)
   
#     print(config["batch_size"])
#     data = ut.STL10Dataset(pathData,config)
#     model= TL.TLModel(num_epochs,config["lr"])
#     training = tm.Training(model.model_resnet18,model.device,
#                        model.loss_function,
#                        model.optimizer,
#                        num_epochs,
#                        data.train_dl,data.val_dl,
#                        sanity_check,model.lr_scheduler,
#                        path2Weights)

#     # result = tune.run(
#     #     partial(training.training_loop),
#     #     config = config,
#     #     num_samples = 10,
#     #     scheduler=scheduler)
    
#     # best_trial = result.get_best_trial("loss", "min", "last")
#     # print(f"Best trial config: {best_trial.config}")
#     # print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
#     # print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

# if __name__ == "__main__":
#     main()