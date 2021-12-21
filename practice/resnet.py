import torchvision.models as models
import torch
import torch.nn as nn
if __name__ == '__main__':
     model = models.resnet50(pretrained=True)