"""Batched execution of the sealed primary models, using unchanged generic layers."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from cognix.adapters.carla import graph_fit_export as ge
from cognix.graph.epistemic_gat import EpistemicGAT


class SharedNodeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 12, bias=False)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(12, 1, bias=False)
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, nodes, epistemic=None, audit_attention=False):
        ge.require(nodes.ndim == 3 and nodes.shape[1:] == (3, 3), "model input [B,3,3]")
        logits = self.output(self.dropout(F.elu(self.hidden(nodes)))).squeeze(-1)
        q = torch.sigmoid(logits).mean(dim=-1)
        return (q, []) if audit_attention else q


class BatchedGAT(nn.Module):
    """Reuse generic W/a/activations/dropout with batch-dimension tensor algebra.

    Tests compare this execution path directly to EpistemicGATLayerPT.forward.
    Raw float64 E computes the Python-float reciprocal before float32 storage,
    matching the frozen generic compute_epistemic_weights, including rounding.
    """
    def __init__(self, use_epistemic_prior):
        super().__init__()
        self.gat = EpistemicGAT(num_layers=2, input_dim=3, hidden_dim=8, output_dim=1,
                               dropout=0.1, use_epistemic_prior=use_epistemic_prior)
        self.use_epistemic_prior = use_epistemic_prior
        self.register_buffer("adjacency", torch.tensor(ge.ADJACENCY, dtype=torch.float32), persistent=False)

    def forward(self, nodes, epistemic=None, audit_attention=False):
        ge.require(nodes.ndim == 3 and nodes.shape[1:] == (3, 3), "model input [B,3,3]")
        e = nodes[:, :, 1].to(torch.float64) if epistemic is None else epistemic
        ge.require(e.shape == nodes.shape[:2] and torch.isfinite(e).all().item()
                   and (e >= 0).all().item(), "canonical epistemic must be finite and nonnegative")
        if self.use_epistemic_prior:
            prior = (1.0 / (1.0 + e.to(torch.float64))).to(nodes.dtype)
        else:
            prior = torch.ones_like(e, dtype=nodes.dtype)
        h, diagnostics = nodes, []
        for index, layer in enumerate(self.gat.layers):
            wh = layer.W(h)
            receiver = wh.unsqueeze(2).expand(-1, -1, 3, -1)
            sender = wh.unsqueeze(1).expand(-1, 3, -1, -1)
            logits = layer.leaky_relu(layer.a(torch.cat((receiver, sender), dim=-1)).squeeze(-1))
            logits = logits + torch.log(prior).unsqueeze(1)
            mask = torch.where(self.adjacency > 0, torch.zeros_like(logits), torch.full_like(logits, -1e9))
            alpha = F.softmax(logits + mask, dim=-1) * self.adjacency
            alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            if audit_attention:
                # Audit normalized incoming weights BEFORE training dropout.
                ge.require(torch.isfinite(alpha).all().item()
                           and torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-6)
                           and not torch.diagonal(alpha, dim1=-2, dim2=-1).any().item(), "attention integrity")
                diagnostics.append(alpha)
            h = torch.bmm(layer.dropout(alpha), wh)
            if index != len(self.gat.layers) - 1:
                h = F.elu(h)
        q = torch.sigmoid(h[:, :, 0]).mean(dim=-1)
        return (q, diagnostics) if audit_attention else q


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def paired_models(seed):
    torch.manual_seed(int(seed))
    nograph = SharedNodeMLP()
    torch.manual_seed(int(seed))
    standard = BatchedGAT(False)
    epistemic = BatchedGAT(True)
    # Explicit tensor cloning; no assumption about two independent initializers.
    epistemic.load_state_dict(copy.deepcopy(standard.state_dict()))
    ge.require(all(torch.equal(v, epistemic.state_dict()[k]) for k, v in standard.state_dict().items()), "paired initialization")
    ge.require(parameter_count(nograph) == 48 and parameter_count(standard) == parameter_count(epistemic) == 50,
               "preregistered parameter counts")
    return {"nograph": nograph, "standard_gat": standard, "epistemic_gat": epistemic}


def model_configuration(method):
    if method == "nograph":
        return {"layers": [3, 12, 1], "bias": False, "activation": "ELU", "hidden_dropout": 0.1,
                "pooling": "mean(node_sigmoid)", "parameter_count": 48}
    ge.require(method in ("standard_gat", "epistemic_gat"), "unknown primary method")
    return {"layers": [3, 8, 1], "heads": 1, "bias": False, "hidden_activation": "ELU",
            "attention_leaky_relu_slope": 0.2, "attention_dropout": 0.1, "dropout_location": "post_softmax",
            "output_activation": "linear", "pooling": "mean(node_sigmoid)", "parameter_count": 50,
            "sender_prior": "1/(1+raw_E)" if method == "epistemic_gat" else "1",
            "prior_strength": 1, "prior_float_semantics": "float64 reciprocal then float32 prior, generic log"}
