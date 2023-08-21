
import torch
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

#wie kann ich diese beiden Parameter aus kedro laden? wäre das sinnvoll?
n_qubits = 10
number_classes=10

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    # strongly entangling layer - weights = {(n_layers , n_qubits, n_parameters)}
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(number_classes)]

class C_layers(nn.Module):
    def __init__(self):
        super(C_layers, self).__init__()
        self.classical_net = nn.Sequential(
            nn.Conv2d(
                1, 8, 3, stride=1, padding=1
            ),  # input size = 1x28x28 -> hidden size = 8x28x28
            nn.Dropout(p=0.5),
            nn.ReLU(True),
            nn.Conv2d(
                8, 16, 3, stride=2, padding=1
            ),  # input size = 8x28x28 -> hidden size = 16x14x14
            nn.Dropout(p=0.3),
            nn.ReLU(True),
            nn.Conv2d(
                16, 32, 3, stride=2, padding=1
            ),  # hidden size = 16x14x14 -> hidden size = 32x7x7
            nn.ReLU(True),
            nn.Conv2d(32, 64, 7),  # hidden size = 32x7x7 -> hidden size = 64x1x1
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(64, 16),  # hidden size = 64 -> hidden size = 16
            nn.Tanh(),
            nn.Linear(16, n_qubits),  # hidden size = 16 -> hidden size = 10x1x1
            nn.Tanh(),
            )
    
    def forward(self, x):
        x = self.classical_net(x.float())
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.clayer = C_layers()
        weight_shapes = {"weights": (1, 10, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, x):
        x = self.clayer(x)
        x = self.qlayer(x)
        return F.softmax(torch.Tensor(x))
