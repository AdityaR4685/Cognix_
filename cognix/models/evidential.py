"""
Evidential Deep Learning / Prior Networks.
Reference: Malinin et al. (2018). Predictive Uncertainty Estimation via Prior Networks.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass

class EvidentialNetwork(nn.Module):
    """
    Network that predicts Dirichlet concentration parameters (alpha) instead of logits.
    """
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Outputs alpha = exp(logits) + 1 to ensure alpha > 1.
        """
        features = self.backbone(x)
        logits = self.head(features)
        
        # Ensure alphas > 1 (Evidential DL constraint)
        alpha = torch.exp(logits) + 1.0
        return alpha

class DirichletLoss(nn.Module):
    """
    Dirichlet Expected Mean Square Error (EDL Loss).
    """
    def __init__(self, num_classes: int, annealing_step: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        
    def forward(self, alphas: torch.Tensor, labels: torch.Tensor, global_step: int = 0) -> torch.Tensor:
        """
        Args:
            alphas: (N, C) predicted concentration parameters
            labels: (N,) true integer labels
        """
        N = alphas.size(0)  # noqa: F841
        y = F.one_hot(labels, num_classes=self.num_classes).float()
        
        S = torch.sum(alphas, dim=1, keepdim=True)
        probs = alphas / S
        
        # EDL Expected MSE Loss
        loss_err = torch.sum((y - probs)**2, dim=1, keepdim=True)
        loss_var = torch.sum(alphas * (S - alphas) / (S**2 * (S + 1)), dim=1, keepdim=True)
        
        # KL Divergence term (regularization)
        annealing_coef = min(1.0, global_step / self.annealing_step)
        
        alpha_tilde = y + (1 - y) * alphas
        S_tilde = torch.sum(alpha_tilde, dim=1, keepdim=True)
        
        kl_div = torch.lgamma(S_tilde) - torch.sum(torch.lgamma(alpha_tilde), dim=1, keepdim=True) + \
                 torch.sum(torch.lgamma(torch.ones_like(alpha_tilde)), dim=1, keepdim=True) - torch.lgamma(torch.ones_like(S_tilde) * self.num_classes)
        
        kl_div += torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde)), dim=1, keepdim=True)
        
        loss = loss_err + loss_var + annealing_coef * kl_div
        return loss.mean()
