import torch
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ansaetze import ansaetze

class QLayers:
    def __init__(self, n_qubits, n_layers, number_classes, data_reuploading=True):
        self.number_classes = number_classes
        if not self.number_classes <= n_qubits:
            raise ValueError(
                f"Number of classes {self.number_classes} may not be higher than number of qubits {n_qubits}"
            )

        self.n_qubits = n_qubits

        self.data_reupload = 1


        self.iec = qml.templates.AngleEmbedding
        self.vqc = ansaetze.circuit_19

        self.argnum = range(self.n_qubits, self.n_qubits + self.n_qubits * n_layers)
        self.weight_shape = {"weights": [n_layers, n_qubits, self.vqc(None)]}
    # def quantum_circuit(self, weights, inputs=None):
    #     if inputs is None:
    #         inputs = self._inputs
    #     else:
    #         self._inputs = inputs
        
    #     # strongly entangling layer - weights = {(n_layers , n_qubits, n_parameters)}
    #     qml.templates.BasicEntanglerLayers(weights, wires=range(self.n_qubits))
    #     return [qml.expval(qml.PauliZ(i)) for i in range(self.number_classes)]

    def quantum_circuit(self, weights, inputs=None):
        if inputs is None:
            inputs = self._inputs
        else:
            self._inputs = inputs
            
        dru = torch.zeros(len(weights))
        dru[:: int(1 / self.data_reupload)] = 1

        for l, l_params in enumerate(weights):
            if l == 0 or dru[l] == 1:
                self.iec(
                    inputs, wires=range(self.n_qubits)
                )  # half because the coordinates already have 2 dims

            self.vqc(l_params)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.number_classes)]
        # return qml.probs(wires=range(self.number_classes))


class CLayers(nn.Module):
    def __init__(self, n_qubits):
        super(CLayers, self).__init__()

        # Bx1x28x28 -> Bx3x14x14
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        # # Bx3x28x28 -> Bx5x14x14
        # self.conv_layer_2 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=3,
        #         out_channels=5,
        #         kernel_size=4,
        #         stride=1,
        #         padding=0,
        #     ),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(kernel_size=2),
        # )
        # fully connected layer, output 10 classes
        self.fc_layer = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=1,
            #     out_channels=3,
            #     kernel_size=3,
            #     stride=1,
            #     padding=1,
            # ),
            nn.Linear(1*28*28, n_qubits),
            nn.Tanh(),
        )

        self.flatten = nn.Flatten()
        # self.out = nn.Linear(3*14*14, n_qubits)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc_layer(x)
        # x = self.conv_layer_2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = self.flatten(x)
        # x = self.out(x)
        return x


class Model(nn.Module):
    def __init__(self, n_qubits, classes, n_layers):
        super(Model, self).__init__()
        self.n_qubits = n_qubits
        self.number_classes = len(classes)
        self.clayer = CLayers(self.n_qubits)
        dev = qml.device("default.qubit", wires=self.n_qubits)
        self.vqc = QLayers(self.n_qubits, n_layers, self.number_classes)
        self.qnode = qml.QNode(self.vqc.quantum_circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, self.vqc.weight_shape)

    def forward(self, x):
        x = self.clayer(x)
        x = self.qlayer(x)
        return x
