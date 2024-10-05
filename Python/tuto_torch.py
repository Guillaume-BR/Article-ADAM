#%% Import packages
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import numpy as np
#%% Import data
training_data = datasets.FashionMNIST(
    root = "data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
# on prendra des batch de 64 images
train_dataloader = DataLoader(training_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# %%
## Define hyperparameters
learning_rate = 0.001
batch_size = 64
epochs = 5
# %%
# Loss function (to optimize)
loss_fn = nn.CrossEntropyLoss() #test with another one ?
# loss_fn = nn.NLLLoss() 
#%%
# Optimizer
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
# %%
# Full implementation
def train_loop(dataloader,model,loss_fn,optimizer,train_error):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred,y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*batch_size + len(X)
            train_error.append(loss)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader,model,loss_fn,test_error,test_accuracy):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct/= size
    test_error.append(test_loss)
    test_accuracy.append(100*correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#%%
### Test with Adam ###
model = NeuralNetwork() # reinitialize the model
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.NLLLoss() #must add a LogSoftmax layer in the NeuralNetwork in order to work
#optimizer = torch.optim.RMSprop(model.parameters(),lr = learning_rate)
optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)
epochs = 10

train_error_adam = []
test_error = []
test_accuracy = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader,model,loss_fn,optimizer,train_error = train_error_adam)
    test_loop(test_dataloader,model,loss_fn,test_error,test_accuracy)
print("Done!")

#%%
### Test with RMSProp ###
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(),lr = learning_rate)
train_error_RMSProp = []
test_error_RMSProp = []
test_accuracy_RMSProp = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader,model,loss_fn,optimizer,train_error = train_error_RMSProp)
    test_loop(test_dataloader,model,loss_fn,test_error = test_error_RMSProp,test_accuracy=test_accuracy_RMSProp)
print("Done!")

#%%
### Test with SDG ###
model = NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)
train_error_SGD = []
test_error_SGD = []
test_accuracy_SGD = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader,model,loss_fn,optimizer,train_error = train_error_SGD)
    test_loop(test_dataloader,model,loss_fn,test_error = test_error_SGD,test_accuracy=test_accuracy_SGD)
print("Done!")
# %%
# Visualisation of the test error
plt.figure()
plt.plot(np.arange(1,11),np.array(test_error),label = "ADAM",color = "r")
plt.plot(np.arange(1,11),np.array(test_error_RMSProp),label = "RMSProp",color = "b")
plt.plot(np.arange(1,11),np.array(test_error_SGD),label = "SGD",color = "g")
plt.legend()
plt.title("Evolution of the test error for each epochs")
plt.xlabel("Epochs")
plt.ylabel("Test Accuracy")
plt.legend()
plt.show()
# %%
# visualisation of the training error (on voit la convergence de la méthode ?)
plt.figure()
plt.plot(np.arange(1,101),np.array(train_error_adam),label = "ADAM",color = "r")
plt.plot(np.arange(1,101),np.array(train_error_RMSProp),label = "RMSProp",color = "b")
plt.plot(np.arange(1,101),np.array(train_error_SGD),label = "SGD",color = "g")
plt.xlabel("Train iteration")
plt.title("Evolution of the training error for each iteration")
plt.ylabel("Train Loss")
plt.legend()
plt.show()
# plt.savefig("./train_error_visu.svg",format='svg') #si on veut sauvegarder l'image
# %%
