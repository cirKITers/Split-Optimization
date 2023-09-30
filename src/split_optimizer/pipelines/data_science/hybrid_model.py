import torch
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class QLayers:
    def __init__(self, n_qubits, n_layers, number_classes):
        self.number_classes = number_classes
        if not self.number_classes <= n_qubits:
            raise ValueError(
                f"Number of classes {self.number_classes} may not be higher than number of qubits {n_qubits}"
            )

        self.n_qubits = n_qubits
        self.argnum = range(self.n_qubits, self.n_qubits + self.n_qubits * n_layers)

    def quantum_circuit(self, weights, inputs=None):
        if inputs is None:
            inputs = self._inputs
        else:
            self._inputs = inputs
        qml.templates.AngleEmbedding(inputs, wires=range(self.n_qubits))
        # strongly entangling layer - weights = {(n_layers , n_qubits, n_parameters)}
        qml.templates.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.number_classes)]


class CLayers(nn.Module):
    def __init__(self, n_qubits):
        super(CLayers, self).__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        x = self.out(x)
        return x


class Model(nn.Module):
    def __init__(
        self, n_qubits, classes, n_layers
    ):
        super(Model, self).__init__()
        self.n_qubits = n_qubits
        self.number_classes = len(classes)
        self.clayer = CLayers(self.n_qubits)
        weight_shapes = {"weights": (n_layers, self.n_qubits)}
        dev = qml.device("default.qubit", wires=self.n_qubits)
        self.vqc = QLayers(self.n_qubits, n_layers, self.number_classes)
        self.qnode = qml.QNode(self.vqc.quantum_circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        # self.closure = qml.metric_tensor(self.qnode, argnum=[1])

    def forward(self, x):
        x = self.clayer(x)
        x = self.qlayer(x)
        return x
