import torch
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ansaetze import ansaetze

class TorchLayer(qml.qnn.TorchLayer):
    def reset_parameters(self):
        pass

class QModule:
    def __init__(self, n_qubits, n_layers, number_classes, data_reupload):
        self.number_classes = number_classes
        if not self.number_classes <= n_qubits:
            raise ValueError(
                f"Number of classes {self.number_classes} may not be higher than number of qubits {n_qubits}"
            )

        self.n_qubits = n_qubits

        self.data_reupload = data_reupload

        self.iec = qml.templates.AngleEmbedding
        self.vqc = ansaetze.circuit_19

        self.argnum = range(self.n_qubits, self.n_qubits + self.n_qubits * n_layers)
        self.weight_shape = {"weights": [n_layers, n_qubits, self.vqc(None)]}

    

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


class PreClassicalModule(nn.Module):
    def __init__(self, n_qubits):
        super(PreClassicalModule, self).__init__()

        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(1 * 28 * 28, n_qubits)
        self.a1 = nn.Tanh()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.a1(x)
        return x

class PostClassicalModule(nn.Module):
    def __init__(self, n_qubits, n_classes):
        super(PostClassicalModule, self).__init__()

        self.l1 = nn.Linear(n_qubits, n_classes)
        self.a1 = nn.ReLU()


    def reset_parameters(self):
        nn.init.xavier_normal_(self.l1.weight)
        nn.init.zeros_(self.l1.bias)

    def forward(self, x):
        x = self.l1(x)
        x = self.a1(x)
        return x

class Model(nn.Module):
    def __init__(self, n_qubits, classes, n_layers, data_reupload):
        super(Model, self).__init__()
        self.n_qubits = n_qubits
        self.number_classes = len(classes)
        self.pre_clayer = PreClassicalModule(self.n_qubits)
        self.post_clayer = PostClassicalModule(self.n_qubits, self.number_classes)
        dev = qml.device("default.qubit", wires=self.n_qubits)
        # self.vqc = QLayers(
        #     self.n_qubits, n_layers, self.number_classes, data_reupload=data_reupload
        # )
        self.vqc = QModule(
            self.n_qubits, n_layers, self.n_qubits, data_reupload=data_reupload
        )
        self.qnode = qml.QNode(self.vqc.quantum_circuit, dev, interface="torch")
        self.qlayer = TorchLayer(self.qnode, self.vqc.weight_shape)

        self.sm = nn.Softmax(dim=1)  # dim=1 because x will have shape bs x n_classes

    def reset_parameters(self):
        """
        This method is intended to be called from the instructor after torch seeds are set
        """
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def q_pre_proc(self, x):
        return (x + 1) * torch.pi / 2

    def forward(self, x):
        # x = (self.clayer(x) + 1) * torch.pi / 2
        # x = self.sm(self.qlayer(x))

        x = self.pre_clayer(x)
        # x = self.q_pre_proc(x)
        # x = self.qlayer(x)
        x = self.post_clayer(x)

        # x = self.sm(x)

        return x
