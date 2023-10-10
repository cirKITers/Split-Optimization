import pennylane as qml


class ansaetze:
    @staticmethod
    def nothing(params):
        pass

    @staticmethod
    def circuit_19(params):
        """
        Generates a single layer of circuit_19

        Args:
            params (torch.tensor|np.ndarray): Parameters that are being utilized in the layer. Expects form to be [n_qubits, n_gates_per_layer], where n_gates_per_layer=3 in this case. If None, then the number of required params per layer per qubit is returned.
        """
        if params is None:
            return 3

        for q, q_params in enumerate(params):
            qml.RX(q_params[0], wires=q)
            qml.RZ(q_params[1], wires=q)

            qml.CRX(q_params[2], wires=[q, (q + 1) % params.shape[0]])
