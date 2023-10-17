import torch
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ansaetze import ansaetze


class TorchLayer(qml.qnn.TorchLayer):
    def reset_parameters(self):
        # nn.init.uniform_(self.qnode_weights['weights'], b=2 * torch.pi)
        nn.init.zeros_(self.qnode_weights["weights"])


class QModule:
    def __init__(self, n_in, n_layers, n_out, data_reupload, disable_learning):
        self.n_out = n_out
        if not self.n_out <= n_in:
            raise ValueError(
                f"Number of classes {self.n_out} may not be higher than number of qubits {n_in}"
            )

        self.n_qubits = n_in
        self.disable_learning = disable_learning

        self.data_reupload = data_reupload

        self.iec = qml.templates.AngleEmbedding
        self.vqc = ansaetze.circuit_19

        if self.disable_learning:
            self.weight_shape = {"weights": []}
            self.argnum = range(self.n_qubits, self.n_qubits)
        else:
            self.weight_shape = {"weights": [n_layers, self.n_qubits * self.vqc(None)]}
            self.argnum = range(
                n_in, n_in + (n_layers * self.n_qubits * self.vqc(None)) + 1
            )

    def quantum_circuit(self, weights, inputs=None):
        if inputs is None:
            inputs = self._inputs
        else:
            self._inputs = inputs

        if self.disable_learning:
            self.iec(inputs, wires=range(self.n_qubits))
        else:
            dru = torch.zeros(len(weights))
            dru[:: int(1 / self.data_reupload)] = 1

            for l, l_params in enumerate(weights):
                if l == 0 or dru[l] == 1:
                    self.iec(
                        inputs, wires=range(self.n_qubits)
                    )  # half because the coordinates already have 2 dims

                self.vqc(l_params)

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_out)]


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
    def __init__(
        self, n_qubits, classes, n_layers, data_reupload, quant_status, n_shots
    ):
        super(Model, self).__init__()
        self.n_qubits = n_qubits
        self.quant_status = quant_status
        self.number_classes = len(classes)
        self.pre_clayer = PreClassicalModule(self.n_qubits)
        self.post_clayer = PostClassicalModule(self.n_qubits, self.number_classes)
        self.n_shots = n_shots

        if self.quant_status == 0:  # passthrough
            self.qlayer = nn.Identity()
        else:
            dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.n_shots)

            self.vqc = QModule(
                self.n_qubits,
                n_layers,
                self.n_qubits,
                data_reupload=data_reupload,
                disable_learning=(
                    self.quant_status == 1
                ),  # either iec only (1) or iec + pqc (2)
            )
            self.qnode = qml.QNode(self.vqc.quantum_circuit, dev, interface="torch")

            self.qlayer = TorchLayer(self.qnode, self.vqc.weight_shape)

        self.sm = nn.Softmax(dim=1)  # dim=1 because x will have shape bs x n_classes

    def get_clayers(self):
        return [self.pre_clayer, self.post_clayer]

    def get_qlayers(self):
        return [self.qlayer]

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

        if self.n_shots is not None: # This super hacky and probably cause qng and spsa to fail!
            meas = []
            for _x in x:
                meas.append(self.qlayer(_x))
            x = torch.stack(meas)
        else:
            self.qlayer(x)
        x = self.post_clayer(x)

        # x = self.sm(x)

        return x
