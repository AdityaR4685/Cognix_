"""
Standard Graph Convolutional Network (GCN) implementation using NumPy.
Reference: Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.

Note: This is a BASELINE method, not a proposed Cognix mechanism.
"""

import numpy as np

class GCNLayer:
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))

    def forward(self, H: np.ndarray, A: np.ndarray) -> np.ndarray:
        """
        Forward pass for GCN Layer.
        H: Node features, shape (num_nodes, in_features)
        A: Adjacency matrix, shape (num_nodes, num_nodes)
        """
        # A_hat = A + I
        I = np.eye(A.shape[0])  # noqa: E741
        A_hat = A + I
        
        # Degree matrix D
        D = np.diag(np.sum(A_hat, axis=1))
        D_inv_sqrt = np.linalg.inv(np.sqrt(D))
        
        # A_norm = D^{-1/2} A_hat D^{-1/2}
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        
        # H' = ReLU(A_norm @ H @ W)
        Z = A_norm @ H @ self.W
        return np.maximum(0, Z)

class GCNNetwork:
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        self.layers = []
        if num_layers == 1:
            self.layers.append(GCNLayer(input_dim, output_dim))
        else:
            self.layers.append(GCNLayer(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            self.layers.append(GCNLayer(hidden_dim, output_dim))

    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """
        Forward pass through the GCN network.
        node_features shape: (num_agents, feature_dim)
        adjacency_matrix shape: (num_agents, num_agents)
        """
        H = node_features
        for layer in self.layers:
            H = layer.forward(H, adjacency_matrix)
        return H
