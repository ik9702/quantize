import os
currentPath = os.getcwd()

import torch
from torch import nn
import numpy as np
from torch import nn
import matplotlib.pyplot as plt

from utils import timestamp
from dataloader import DS
from torch.utils.data import DataLoader
from model import AE


train_dataset = DS(train=True)
test_dataset = DS(train=False)

batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)






device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model = AE()
model = model.to(device)



optim = torch.optim.AdamW(params = model.parameters(), lr=1e-3)
loss_function = nn.L1Loss()








from torch.utils.tensorboard import SummaryWriter

logdir = f"output/{timestamp()}"
boardir = f"{logdir}/board"
ckptdir = f"{logdir}/ckpt"
os.makedirs(logdir)
os.makedirs(boardir)
os.makedirs(ckptdir)
writer = SummaryWriter(log_dir=boardir)
tot_lp = 0


lp = 1000
train_loss = 0
for epoch in range(1,lp+1):
    print(f"epoch : {epoch}")
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_function(pred, y)
        train_loss += loss.item()
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if((batch+1) % 100 == 0) and batch != 0:
            print(f"{batch+1}/{len(train_dataloader)} : train_gloss : {train_loss / (batch+1)}")

    train_loss = train_loss / len(train_dataloader.dataset)
    
    test_loss = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_function(pred, y)
            test_loss += loss.item()
            
    test_loss = test_loss / len(test_dataloader.dataset)

    print(f"epoch {epoch} Done!  |  test_gloss : {test_loss}")
    writer.add_scalars('gloss', {'train' : train_loss,
                               'test' : test_loss}, tot_lp)  
    
    
    if(epoch%5==1):
        torch.save(model.state_dict(), f"{ckptdir}/ckpot{tot_lp}_{test_loss:.4f}.pt")
        model.eval()
    
    writer.flush()
    tot_lp += 1
writer.close()
