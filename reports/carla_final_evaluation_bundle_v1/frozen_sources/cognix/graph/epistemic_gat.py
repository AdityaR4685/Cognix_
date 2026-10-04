"""
Epistemic-Aware Graph Attention Network (EpistemicGAT) — PyTorch implementation.

DESIGN INTENT:
    This is a TRAINED neural network. Its weight matrices W and attention
    vector a are optimized on training data. This makes it consistent with
    the COGNIX claim that M4 produces "refined agent representations."

    Without training, the random projection W produces arbitrary outputs.
    The epistemic prior on attention is the novel COGNIX contribution; the
    learned representation is required to make the pipeline non-trivial.

PROPOSED COGNIX MECHANISM:
    Attention prior: w_initial(j->i) = 1 / (1 + sigma_e_j)
    This acts as an additive log-prior on the standard GAT attention logits:
        e_ij_epistemic = e_ij_standard + log(w_initial(j->i))

    Agents with high epistemic uncertainty exert less influence on neighbors.
    This is a proposed mechanism requiring experimental validation.

TRAINING PROTOCOL:
    - Train on TRAINING data only (X_train, y_train).
    - Input node features: [p_i, sigma_e_i, sigma_a_i] per agent.
    - Output: refined per-agent probability p_i_refined = sigmoid(H_prime[i, 0]).
    - Collective loss: BCE(mean(p_i_refined), y) — collective agreement.
    - Agent ordering is fixed by agent_id sorted alphabetically.

KNOWN LIMITATIONS:
    1. The epistemic prior (sigma_e ≈ 0.001-0.004 for well-trained agents)
       produces near-uniform weights (contrast < 0.4%). The prior's effect
       is only visible under severe degradation (sigma_e >> 0.1).
    2. GAT is not trained with the final calibration pipeline in the loop.
       This is a limitation of the current training protocol.
    3. For the 4-agent toy experiment, 2^4=16 Shapley subsets are feasible.
       For publication, the effect of epistemic prior vs learned attention
       should be ablated separately.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from cognix.core.interfaces import GraphRefinement
from cognix.core.types import GraphResult

class EpistemicGATLayerPT(nn.Module):
    """
    Single graph attention layer with epistemic uncertainty prior.
    Implemented in PyTorch for end-to-end training.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Learned linear projection (Xavier-uniform init by default via kaiming)
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector (operates on concatenated pair)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(
        self,
        H: torch.Tensor,             # (N, in_features)
        A: torch.Tensor,             # (N, N) binary adjacency, 1=edge
        epistemic_prior: torch.Tensor,  # (N, N) column j = 1/(1+sigma_e_j)
        final_layer: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = H.shape[0]
        Wh = self.W(H)                             # (N, out_features)

        # Pairwise concatenation for attention score
        Whi = Wh.unsqueeze(1).expand(-1, N, -1)   # (N, N, out_features)
        Whj = Wh.unsqueeze(0).expand(N, -1, -1)   # (N, N, out_features)
        pair = torch.cat([Whi, Whj], dim=-1)       # (N, N, 2*out_features)

        # Raw attention logits: e_ij = a^T [Wh_i || Wh_j]
        e = self.leaky_relu(self.a(pair).squeeze(-1))   # (N, N)

        # COGNIX epistemic prior: additive log-prior to suppress attention of uncertain senders
        # e_ij_epistemic = e_ij + log(1/(1+sigma_e_j))
        e_epistemic = e + torch.log(epistemic_prior)

        # Mask non-edges to large negative (no artificial +1 bias)
        mask = torch.where(A > 0, torch.zeros_like(e), torch.full_like(e, -1e9))
        e_masked = e_epistemic + mask

        # Numerically stable softmax over neighbors
        alpha = F.softmax(e_masked, dim=-1)        # (N, N)
        # Zero out non-edges exactly (prevent numerical leakage)
        alpha = alpha * A
        row_sums = alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        alpha = alpha / row_sums                   # re-normalize after masking

        alpha = self.dropout(alpha)

        # Aggregate: H' = alpha @ Wh
        H_agg = alpha @ Wh                         # (N, out_features)

        if final_layer:
            # Linear output: preserve full value range for sigmoid extraction
            H_out = H_agg
        else:
            # Non-linear activation for intermediate layers
            H_out = F.elu(H_agg)

        return H_out, alpha


class EpistemicGAT(nn.Module, GraphRefinement):
    """
    Full Epistemic-Aware GAT — trainable PyTorch module.

    Parameters
    ----------
    num_layers : int   — number of GAT layers
    input_dim  : int   — input node feature dimension (typically 3: [p, epi, ale])
    hidden_dim : int   — hidden layer width
    output_dim : int   — output node feature dimension

    Architecture decision:
        intermediate layers: ELU activation (non-zero-saturating unlike ReLU)
        final layer:         linear (no activation) — allows sigmoid extraction
                             without zero-collapse from ReLU

    Training:
        optimizer = Adam, lr=1e-3
        loss      = BCE(mean(sigmoid(H_prime[:,0])), y)
        epochs    = configurable (default 100)
        data      = TRAINING SET ONLY (X_train, y_train)
        ordering  = agents sorted by agent_id (deterministic)
    """

    def __init__(
        self,
        num_layers: int = 2,
        input_dim: int = 3,
        hidden_dim: int = 8,
        output_dim: int = 4,
        dropout: float = 0.1,
        use_epistemic_prior: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_epistemic_prior = use_epistemic_prior
        self._trained = False

        layers = []
        if num_layers == 1:
            layers.append(EpistemicGATLayerPT(input_dim, output_dim, dropout))
        else:
            layers.append(EpistemicGATLayerPT(input_dim, hidden_dim, dropout))
            for _ in range(num_layers - 2):
                layers.append(EpistemicGATLayerPT(hidden_dim, hidden_dim, dropout))
            layers.append(EpistemicGATLayerPT(hidden_dim, output_dim, dropout))
        self.layers = nn.ModuleList(layers)

    def compute_epistemic_weights(
        self,
        epistemic_uncertainties: dict[str, float],
        agent_order: list[str],
    ) -> np.ndarray:
        """
        W_prior[i, j] = 1 / (1 + sigma_e_j)

        Column j is the attention-reduction weight for agent j as a sender.
        Higher sigma_e_j -> lower w_prior -> less attention from j to neighbors.
        """
        N = len(agent_order)
        W_prior = np.zeros((N, N), dtype=np.float32)
        for j_idx, j_agent in enumerate(agent_order):
            if self.use_epistemic_prior:
                sigma_e_j = epistemic_uncertainties.get(j_agent, 0.0)
                W_prior[:, j_idx] = 1.0 / (1.0 + sigma_e_j)
            else:
                W_prior[:, j_idx] = 1.0
        return W_prior

    def forward(
        self,
        node_features: np.ndarray,              # (N, input_dim)
        adjacency_matrix: np.ndarray,           # (N, N)
        epistemic_uncertainties: dict[str, float],
        agent_order: list[str],
    ) -> GraphResult:
        """
        Full forward pass with epistemic attention prior.
        """
        W_prior_np = self.compute_epistemic_weights(
            epistemic_uncertainties, agent_order
        )

        device = next(self.parameters()).device
        H = torch.tensor(node_features, dtype=torch.float32, device=device)
        A = torch.tensor(adjacency_matrix, dtype=torch.float32, device=device)
        ep = torch.tensor(W_prior_np, dtype=torch.float32, device=device)

        attn_list = []
        self.eval()
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                is_final = (i == len(self.layers) - 1)
                H, attn = layer(H, A, ep, final_layer=is_final)
                attn_list.append(attn.cpu().numpy())

        return GraphResult(
            node_outputs=H.cpu().numpy(),
            attention=attn_list,
            metadata={"epistemic_prior_used": self.use_epistemic_prior}
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        agents: list,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        verbose: bool = False,
        seed: int = 42,
    ) -> list[float]:
        """
        Train the GAT on the training set.

        Protocol:
            For each training sample x_i:
              1. Collect per-agent predictions p_i and uncertainties.
              2. Build node features [p_i, sigma_e_i, sigma_a_i].
              3. Forward pass through GAT.
              4. p_collective = mean(sigmoid(H_prime[:, 0]))
              5. Loss = BCE(p_collective, y_i)

        Trains on X_train ONLY. No calibration or test data is used.
        Agent ordering is sorted by agent_id (deterministic).

        Parameters
        ----------
        agents     : list of agent objects with .predict() and .estimate_uncertainty()
        X_train    : (N_train, F) training features
        y_train    : (N_train,) binary labels
        epochs     : number of training epochs
        lr         : Adam learning rate
        verbose    : print loss every 10 epochs
        seed       : random seed for reproducibility
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Sort agents by agent_id for deterministic ordering
        sorted_agents = sorted(agents, key=lambda a: a.agent_id)
        agent_ids = [a.agent_id for a in sorted_agents]
        N = len(sorted_agents)

        # Fully connected adjacency (no self-loops)
        A_np = (np.ones((N, N)) - np.eye(N)).astype(np.float32)
        A_t = torch.tensor(A_np, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        losses = []

        # Precompute agent outputs (since agents are frozen)
        precomputed_nf = []
        precomputed_ep = []
        for x_i in X_train:
            uncs_i = []
            ep_dict = {}
            for agent in sorted_agents:
                p_res = agent.predict(x_i)
                p = float(p_res.value) if hasattr(p_res, 'value') else float(p_res)
                uq = agent.estimate_uncertainty(x_i)
                uncs_i.append([p, uq.epistemic, uq.aleatoric])
                ep_dict[agent.agent_id] = uq.epistemic
            precomputed_nf.append(torch.tensor(uncs_i, dtype=torch.float32))
            ep_np = self.compute_epistemic_weights(ep_dict, agent_ids)
            precomputed_ep.append(torch.tensor(ep_np, dtype=torch.float32))

        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            perm = np.random.permutation(len(X_train))

            for idx in perm:
                y_i = float(y_train[idx])

                nf = precomputed_nf[idx]
                ep_t = precomputed_ep[idx]

                # Move tensors to device (if using GPU, usually CPU here)
                nf = nf.to(next(self.parameters()).device)
                ep_t = ep_t.to(next(self.parameters()).device)
                A_t_device = A_t.to(next(self.parameters()).device)
                
                # Forward pass
                self.train()
                optimizer.zero_grad()
                H = nf
                for i_l, layer in enumerate(self.layers):
                    is_final = (i_l == len(self.layers) - 1)
                    H, _ = layer(H, A_t_device, ep_t, final_layer=is_final)

                # p_i_refined = sigmoid(H[i, 0])
                # p_collective = mean over agents
                p_agents = torch.sigmoid(H[:, 0])          # (N,)
                p_collective = p_agents.mean().unsqueeze(0) # (1,)
                target = torch.tensor([y_i], dtype=torch.float32)

                loss = criterion(p_collective, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(X_train)
            losses.append(avg_loss)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"    GAT epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}")

        self._trained = True
        return losses

    def is_trained(self) -> bool:
        return self._trained

    def save(self, filepath: str) -> None:
        """Save trained model weights to file."""
        from cognix.artifacts.persistence import save_epistemic_gat
        save_epistemic_gat(self, filepath)

    def load(self, filepath: str) -> "EpistemicGAT":
        """Load trained model weights from file."""
        from cognix.artifacts.persistence import load_epistemic_gat
        loaded = load_epistemic_gat(filepath, expected_input_dim=self.input_dim, expected_output_dim=self.output_dim)
        self.load_state_dict(loaded.state_dict())
        self._trained = True
        self.eval()
        return self
