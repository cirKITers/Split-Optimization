import torch
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

TRAINING_SIZE = 1000
TEST_SIZE = 250

train_data = np.load("data/01_raw/Train_[0,1,3,6].npz")  # enthält 7500 samples
test_data = np.load("data/01_raw/Test_[0,1,3,6].npz")  # enthält 1500 samples

x_train = train_data["features"][:TRAINING_SIZE]
x_test = test_data["features"][:TEST_SIZE]

# one-hot-encoding for the labels
y_train = np.zeros((TRAINING_SIZE, len(train_data["classes"])))
y_test = np.zeros((TEST_SIZE, len(train_data["classes"])))

for i in range(TRAINING_SIZE):
    y_train[i, list(train_data["classes"]).index(train_data["labels"][i])] = 1

for i in range(TEST_SIZE):
    y_test[i, list(test_data["classes"]).index(test_data["labels"][i])] = 1

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).float()
dataset = [x_train, y_train, x_test, y_test]


dev = qml.device("default.qubit", wires=6)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(6))
    # strongly entangling layer - weights = {(n_layers , n_qubits, n_parameters)}
    qml.templates.StronglyEntanglingLayers(weights, wires=range(6))
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 6)
        weight_shapes = {"weights": (1, 6, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qlayer(x)
        return F.softmax(torch.Tensor(x))


model = Net()
loss_func = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

loss_list = []
for epoch in range(epochs):
    total_loss = []
    for i in range(TRAINING_SIZE):
        optimizer.zero_grad()
        output = model(dataset[0][i])
        loss = loss_func(output, dataset[1][i])
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(
        100. * (epoch + 1) / epochs, loss_list[-1]))
    

plt.plot(loss_list)
plt.title('Hybrid NN Training Convergence')
plt.xlabel('Training Iterations')
plt.ylabel('Neg Log Likelihood Loss')
plt.savefig("plot.png")

model.eval()
with torch.no_grad():   
    correct = 0
    for i in range(TEST_SIZE):
        output = model(dataset[2][i])
        
        pred = output.argmax() 
        if pred == dataset[3][i].argmax():
            correct += 1
        
        loss = loss_func(output, dataset[3][i])
        total_loss.append(loss.item())
        
    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss) / len(total_loss),
        correct / TEST_SIZE * 100)
        )