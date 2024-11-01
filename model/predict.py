import json
import torch
import numpy as np
import torch.nn as nn
from model import nnue,device

[K,A,B,N,R,C,P] = [1,2,3,4,5,6,7]
[k,a,b,n,r,c,p] = [-1,-2,-3,-4,-5,-6,-7]

init_game_board = np.asarray([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,r,n,b,a,k,a,b,n,r,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,c,0,0,0,0,0,c,0,0,0,0],
    [0,0,0,p,0,p,0,p,0,p,0,p,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,P,0,P,0,P,0,P,0,P,0,0,0],
    [0,0,0,0,C,0,0,0,0,0,C,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,R,N,B,A,K,A,B,N,R,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
])

[red,black] = [1,-1]

def convert_board_to_input(board,now_go_side,to_tensor=True):
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
    input_data = np.array(input_data, dtype=np.float32)
    if to_tensor:
        input_data = torch.from_numpy(input_data).to(device)
    return input_data

def predict(model_dict_path):
    model = nnue().to(device)
    model.load_state_dict(torch.load(model_dict_path))
    model.eval()
    input_data = convert_board_to_input(init_game_board,red)
    y = model(input_data)
    y = torch.softmax(y,dim=0)
    print(y.cpu().detach().numpy(),float(y[2] - y[0]))

if __name__ == "__main__":
    predict(model_dict_path="save\\epoch_0_model_with_tuc_0.5978")