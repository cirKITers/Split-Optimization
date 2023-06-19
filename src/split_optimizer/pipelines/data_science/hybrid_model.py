
import torch
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F



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
