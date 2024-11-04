import torch
import torch.nn as nn
import torchsummary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class nnue_classifier(nn.Module):
    def __init__(self,input_size = 256 * 7 + 1):
        super().__init__()
        self.fc_1 = nn.Linear(input_size,128)
        self.fc_2 = nn.Linear(128,32)
        self.fc_3 = nn.Linear(32, 32)
        self.fc_4 = nn.Linear(32, 3)
        self.relu = nn.ReLU()
    def forward(self,x):
        y = self.fc_1(x)
        y = self.relu(y)
        y = self.fc_2(y)
        y = self.relu(y)
        y = self.fc_3(y)
        y = self.relu(y)
        y = self.fc_4(y)
        return y

class nnue_regression(nn.Module):
    def __init__(self,input_size = 256 * 7 + 1):
        super().__init__()
        self.fc_1 = nn.Linear(input_size,256)
        self.fc_2 = nn.Linear(256,32)
        self.fc_3 = nn.Linear(32, 32)
        self.fc_4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    def forward(self,x):
        y = self.fc_1(x)
        y = self.relu(y)
        y = self.fc_2(y)
        y = self.relu(y)
        y = self.fc_3(y)
        y = self.relu(y)
        y = self.fc_4(y)
        return y

if __name__ == "__main__":
    model = nnue_classifier(input_size=90 * 7 + 1).to(device)
    torchsummary.summary(model,input_size=(2,90 * 7 + 1),device=device)
    print("---------------------------------------------------------------")
    model = nnue_regression(input_size=90 * 7 + 1).to(device)
    torchsummary.summary(model, input_size=(2, 90 * 7 + 1), device=device)