import json
import random

import numpy as np
import torch
import torch.nn as nn
import os
from model import nnue,device
from sklearn.metrics import roc_auc_score

def get_filepaths(directory,extension="json"):
    filepaths = []
    for root,dirs,files in os.walk(directory):
        for _file in files:
            if _file.endswith(extension):
                filepaths.append(os.path.join(root,_file))
    print("size of filepaths = ",len(filepaths))
    return filepaths

def convert_data_to_tensor(train_epoch_filepaths,to_tensor=True):
    input_list = []
    output_list = []
    for path in train_epoch_filepaths:
        json_data = json.load(open(path))
        for single_data in json_data:
            board = single_data['board']
            now_go_side = single_data['now_go_side']
            win_side = single_data['win_side']
            #
            input_data = np.zeros(shape=(256 * 7 + 1))
            input_data[-1] = now_go_side  # 1 or -1
            pos = 0
            for line in board:
                for piece in line:
                    if piece != 0:
                        convert_piece = abs(piece) - 1
                        convert_pos = convert_piece * 256 + pos
                        input_data[convert_pos] = 1 if piece > 0 else -1
                    pos += 1

            input_list.append(input_data)
            #
            output_data = win_side + 1
            output_list.append(output_data)
            #print(output_data)
    input_list = np.array(input_list,dtype=np.float32)
    output_list = np.array(output_list,dtype=np.int64)
    if to_tensor:
        input_list = torch.from_numpy(input_list).to(device)
        output_list = torch.from_numpy(output_list).to(device)
    return input_list,output_list

def train(converted_data_path):
    filepaths = get_filepaths(converted_data_path)
    split_spot = int(len(filepaths) * 0.95)
    train_filepaths = filepaths[:split_spot]
    test_filepaths = filepaths[split_spot:]
    #
    random.seed(6666)
    random.shuffle(train_filepaths)
    random.shuffle(test_filepaths)
    batch_size = 32
    #
    train_batch_sum = len(train_filepaths) // batch_size
    test_batch_sum = len(test_filepaths) // batch_size
    #
    model = nnue().to(device)
    loss_method = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.RAdam(model.parameters(),lr=3e-4)
    for epoch in range(10000):
        train_loss = 0
        train_acc = 0
        for train_batch in range(train_batch_sum):
            start = batch_size * train_batch
            end = min(batch_size * (train_batch + 1),len(train_filepaths))
            train_epoch_filepaths = train_filepaths[start:end]
            train_input_list,train_output_list = convert_data_to_tensor(train_epoch_filepaths)
            #
            y = model(train_input_list)
            train_acc += (y.argmax(1) == train_output_list).sum() / len(train_output_list)
            loss = loss_method(y,train_output_list)
            train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            #
            if train_batch % 100 == 0:
                print(f"epoch {epoch} | batch {train_batch} : {train_batch_sum} | train_loss {train_loss / (train_batch + 1)} | train_acc {train_acc / (train_batch + 1)}")
        test_loss = 0
        test_acc = 0
        test_entire_output_list = []
        test_entire_predict_list = []
        for test_batch in range(test_batch_sum):
            start = batch_size * test_batch
            end = min(batch_size * (test_batch + 1),len(test_filepaths))
            test_epoch_filepaths = test_filepaths[start:end]
            test_input_list,test_output_list = convert_data_to_tensor(test_epoch_filepaths)
            y = model(test_input_list)
            loss = loss_method(y, test_output_list)
            test_acc += (y.argmax(1) == test_output_list).sum() / len(test_output_list)
            #
            y = torch.softmax(y,dim=1)
            test_entire_output_list.extend(test_output_list.tolist())
            test_entire_predict_list.extend(y.cpu().detach().tolist())
            test_loss += loss.item()
            if test_batch % 100 == 0:
                print(f"epoch {epoch} | batch {test_batch} : {test_batch_sum} | test_loss {test_loss / (test_batch + 1)} | test_acc {test_acc / (test_batch + 1)}")
        test_auc = roc_auc_score(y_true=test_entire_output_list,y_score=test_entire_predict_list,multi_class='ovo')
        print(f"epoch {epoch} | test auc {test_auc}")
        torch.save(model.state_dict(),f"./save/epoch_{epoch}_model_with_tuc_{round(test_auc,4)}")

if __name__ == "__main__":
    train(converted_data_path="../dump")