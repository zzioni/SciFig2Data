import torch
import time
import copy
from tqdm import tqdm
import os
import numpy as np


def train_model(model,
                dataloaders,
                criterion,
                optimizer,
                device,
                output_dir,
                num_epochs=25,
                class_acc_dic : dict = {}):
    
    since = time.time()

    train_acc_history = []
    val_acc_history = []
    
    acc_per_class = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_class = len(class_acc_dic.keys())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        with open(os.path.join(output_dir, "result.txt"), "a") as f:
            f.write('Epoch {}/{}\n'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            corrects_dic_per_class = {i:0 for i in range(num_class)}
            labelnum_dic_per_class = {i:0 for i in range(num_class)}

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # statistics - per class
                for i in range(num_class):
                    con_a = preds == labels.data
                    con_b = labels.detach().cpu() == torch.tensor([i]*len(preds))
                    con = np.logical_and(np.array(con_a.detach().cpu()),np.array(con_b))
                    corrects_dic_per_class[i] += torch.sum(torch.tensor(con))
                    labelnum_dic_per_class[i] += torch.sum(con_b)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            with open(os.path.join(output_dir, "result.txt"), "a") as f:
                f.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(output_dir, 'best.pt'))
                
                # accuracy per class
                for i in range(num_class):
                    acc_per_class = corrects_dic_per_class[i].double() / labelnum_dic_per_class[i].double()
                    class_acc_dic[i] = acc_per_class.data
                    
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                torch.save(model.state_dict(), os.path.join(output_dir, 'last.pt'))
            if phase == 'train':
                train_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    with open(os.path.join(output_dir, "result.txt"), "a") as f:
        f.write('Best val Acc: {:4f}\n'.format(best_acc))
        f.write('Val accuracy per Class: {}'.format(str(class_acc_dic)))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, val_acc_history