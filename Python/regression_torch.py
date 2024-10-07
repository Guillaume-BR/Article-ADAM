#%%
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

#%%
# Import data
data = fetch_california_housing()
print(data.feature_names)
X, y = data.data, data.target

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#%%
# Define Neural Network
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8,24),
            nn.ReLU(),
            nn.Linear(24,12),
            nn.ReLU(),
            nn.Linear(12,6),
            nn.ReLU(),
            nn.Linear(6,1)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        ajust = self.linear_relu_stack(x)
        return ajust


X_train_raw, X_test_raw, y_train, y_test = train_test_split(X,y,train_size = 0.7,shuffle = True)
scaler = StandardScaler()
scaler.fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

batch_size = 10
batch_start = torch.arange(0,len(X_train),batch_size)

best_mse = np.inf
best_weight = None
learning_rate = 1e-3
n_epochs = 100
# %%
### Test with Adam ###
history_train_adam = []
history_test_adam = []
model = MyModel().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start,unit="batch",mininterval=0,disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            y_pred = model(X_batch)
            
            loss = loss_fn(y_pred,y_batch)
            if float(start)%1000 == 0:
                history_train_adam.append(float(loss))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            bar.set_postfix(mse = float(loss))
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred,y_test)
    mse = float(mse)
    history_test_adam.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
    
model.load_state_dict(best_weights)
print("MSE Adam: %.2f" % best_mse)
print("RMSE Adam: %.2f" % np.sqrt(best_mse))


#%%
### Test with RMSProp
history_train_rmsprop = []
history_test_rmpsprop = []
model = MyModel().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(),lr=learning_rate)

for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start,unit="batch",mininterval=0,disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred,y_batch)
            if float(start)%1000 == 0:
                history_train_rmsprop.append(float(loss))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            bar.set_postfix(mse = float(loss))
    model.eval()
    y_pred = model(X_test)
    mse = loss_fn(y_pred,y_test)
    mse = float(mse)
    history_test_rmpsprop.append(mse)
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
    
model.load_state_dict(best_weights)
print("MSE RMSprop: %.2f" % best_mse)
print("RMSE RMSprop: %.2f" % np.sqrt(best_mse))

# %%
plt.figure()
plt.plot(history_test_adam,label = "Adam", color = "r")
plt.plot(history_test_rmpsprop,label = "RMSprop",color = "b")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.show()
# %%
plt.figure()
plt.plot(history_train_adam,label = "Adam", color = "r")
#plt.plot(history_train_rmsprop,label = "RMSprop",color = "b")
plt.legend()
plt.xlabel("Batch")
plt.ylabel("MSE")
plt.show()
# %%
