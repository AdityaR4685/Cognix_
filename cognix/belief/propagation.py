import numpy as np
from .base import BeliefState

class BeliefPropagator:
    """
    Bayesian Belief Propagation
    Reference: Pearl, J. (1988). Probabilistic Reasoning in Intelligent Systems. Morgan Kaufmann.
    Loopy Belief Propagation: Approximate inference for graphs with cycles.
    """
    def __init__(self, adjacency_matrix: np.ndarray, agent_ids: list[str]):
        self.adj = adjacency_matrix
        self.agent_ids = agent_ids
        self.n = len(agent_ids)
        self.edges = [(i, j) for i in range(self.n) for j in range(self.n) if self.adj[i, j] > 0]
        # Initialize messages: uniform distribution initially
        self.messages = {}
        for (i, j) in self.edges:
            self.messages[(i, j)] = None  # Will be initialized based on shape

    def propagate(self, beliefs: dict[str, BeliefState], max_iterations: int = 5, 
                  convergence_threshold: float = 1e-4, damping: float = 0.5) -> dict[str, BeliefState]:
        
        # Convert beliefs dict to list matching agent_ids order
        local_beliefs = []
        for aid in self.agent_ids:
            local_beliefs.append(beliefs[aid].belief)
            
        n_classes = len(local_beliefs[0])
        
        # Initialize messages if not done
        for (i, j) in self.edges:
            if self.messages[(i, j)] is None or len(self.messages[(i, j)]) != n_classes:
                self.messages[(i, j)] = np.ones(n_classes) / n_classes

        for iteration in range(max_iterations):
            new_messages = {}
            max_diff = 0.0
            
            for (i, j) in self.edges:
                # Compute message m_ij = product of incoming messages to i except from j * local belief i
                m_ij = np.copy(local_beliefs[i])
                
                # Multiply incoming messages to i from k != j
                for k in range(self.n):
                    if (k, i) in self.edges and k != j:
                        m_ij *= self.messages[(k, i)]
                
                # Normalize
                sum_m = np.sum(m_ij)
                if sum_m > 0:
                    m_ij /= sum_m
                else:
                    m_ij = np.ones(n_classes) / n_classes
                    
                # Damped update: m_new = damping * m_old + (1-damping) * m_computed
                m_old = self.messages[(i, j)]
                m_new = damping * m_old + (1.0 - damping) * m_ij
                new_messages[(i, j)] = m_new
                
                # Check convergence (KL divergence approximate max diff)
                diff = np.max(np.abs(m_new - m_old))
                if diff > max_diff:
                    max_diff = diff
                    
            self.messages.update(new_messages)
            
            if max_diff < convergence_threshold:
                break
                
        # Compute final updated beliefs
        updated_beliefs = {}
        for i, aid in enumerate(self.agent_ids):
            b_i = np.copy(local_beliefs[i])
            for k in range(self.n):
                if (k, i) in self.edges:
                    b_i *= self.messages[(k, i)]
            
            sum_b = np.sum(b_i)
            if sum_b > 0:
                b_i /= sum_b
            else:
                b_i = np.ones(n_classes) / n_classes
                
            updated_beliefs[aid] = BeliefState(
                agent_id=aid,
                belief=b_i,
                alpha=beliefs[aid].alpha,
                beta_param=beliefs[aid].beta_param,
                confidence=beliefs[aid].confidence
            )
            
        return updated_beliefs

    def compute_bethe_free_energy(self, beliefs: dict[str, BeliefState], messages: dict[tuple[int, int], np.ndarray]) -> float:
        """
        Reference for Bethe free energy: 
        Yedidia, J. S., Freeman, W. T., & Weiss, Y. (2001). 
        Understanding Belief Propagation and its Generalizations.
        """
        # Simplified placeholder for Bethe free energy computation.
        # Computing the exact BFE requires pairwise marginals which we approximate.
        free_energy = 0.0
        # Energy terms
        for aid in self.agent_ids:
            b = beliefs[aid].belief
            free_energy -= np.sum(b * np.log(np.maximum(b, 1e-12))) # Node entropy
            
        return float(free_energy)
