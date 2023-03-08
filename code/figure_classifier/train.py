import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

import trainer
import model
import dataloader

'''
BASELINE CODE
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="docfigure", help='docfigure or compoundfigure')
    parser.add_argument('--data_dir', type=str, default="/data0/jiyeong/DocFigure/", help='data path')
    parser.add_argument('--output_dir', type=str, help='output path')
    parser.add_argument('--model_name', type=str, default='resnet', help='model name str')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--feature_extract', default=True, help='When False, we finetune the whole model, when True we only update the reshaped layer params')
    opt = parser.parse_args()
    
    # Make Output Dir
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # labelnames
    labelNames_docfigure = ['3D objects',
    'Algorithm',
    'Area chart',
    'Bar plots',
    'Block diagram',
    'Box plot',
    'Bubble Chart',
    'Confusion matrix',
    'Contour plot',
    'Flow chart',
    'Geographic map',
    'Graph plots',
    'Heat map',
    'Histogram',
    'Mask',
    'Medical images',
    'Natural images',
    'Pareto charts',
    'Pie chart',
    'Polar plot',
    'Radar chart',
    'Scatter plot',
    'Sketches',
    'Surface plot',
    'Tables',
    'Tree Diagram',
    'Vector plot',
    'Venn Diagram']
    
    labelNames_compoundfigure = ['Compound figure', 'Single figure']
    
    labelNames = labelNames_docfigure if opt.dataset == 'docfigure' else labelNames_compoundfigure
    
    class_acc_dic = {c:0.0 for c in range(len(labelNames))}
    
    # Initialize the model for this run
    model_ft, input_size = model.initialize_model(opt.model_name, len(labelNames), opt.feature_extract, use_pretrained=True)

    # Print the model we just instantiated
    print(model_ft)
    
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Dataset Loading
    print("Initializing Datasets and Dataloaders...")
    
    train_dataset = dataloader.FigureClassificationDataset(opt.data_dir, labelnames = labelNames, train=True, transforms = data_transforms['train'])
    test_dataset = dataloader.FigureClassificationDataset(opt.data_dir, labelnames = labelNames, train=False, transforms = data_transforms['test'])

    train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    
    dataloaders_dict = {"train":train_dataloader, "val":test_dataloader}
    
    # Detect if we have a GPU available
    device = torch.device(opt.device if opt.device == 'cpu' else int(opt.device))
    
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are 
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if opt.feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    
    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, train_hist, val_hist = trainer.train_model(model_ft, dataloaders_dict, criterion,
                                                         optimizer_ft, device=device, output_dir=opt.output_dir,
                                                         num_epochs=opt.epochs, class_acc_dic = class_acc_dic)

    # Evaluate : Plot
    train_hist = [h.cpu().numpy() for h in train_hist]
    val_hist = [h.cpu().numpy() for h in val_hist]

    plt.title("Train/Validation Accuracy per Epochs : {}".format(opt.model_name))
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy")
    plt.plot(range(1,opt.epochs+1),train_hist,label="Train Acc")
    plt.plot(range(1,opt.epochs+1),val_hist,label="Validation Acc")
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, opt.epochs+1, 1.0))
    plt.legend()
    plt.savefig(os.path.join(opt.output_dir, 'train_val_acc.png'))   # save the figure to file