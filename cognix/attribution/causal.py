"""
Structural Causal Models (SCMs) and Counterfactual Reasoning.
"""
from typing import Dict, Any, Callable

class StructuralCausalModel:
    """
    Basic Directed Acyclic Graph (DAG) for Causal Inference and Counterfactual generation.
    Nodes are variables, edges define causal dependencies via functions.
    """
    def __init__(self):
        self.nodes = {}          # node_name -> default_value or function
        self.edges = {}          # child -> list of parents
        self.functions = {}      # child -> function(parents)

    def add_node(self, name: str, default: Any = None):
        if name not in self.nodes:
            self.nodes[name] = default
            self.edges[name] = []

    def add_edge(self, parent: str, child: str):
        if child not in self.edges:
            self.edges[child] = []
        if parent not in self.edges[child]:
            self.edges[child].append(parent)

    def set_mechanism(self, node: str, mechanism: Callable):
        """Set the structural equation f(parents) for a node."""
        self.functions[node] = mechanism

    def forward(self, evidence: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solve the SCM top-down. 
        evidence overrides node values (conceptually similar to do-operator for root nodes).
        """
        state = {}
        evidence = evidence or {}
        
        # Simple topological sort assuming DAG
        import networkx as nx
        G = nx.DiGraph()
        for c, parents in self.edges.items():
            for p in parents:
                G.add_edge(p, c)
                
        # Include disconnected nodes
        for n in self.nodes:
            G.add_node(n)
            
        topo_order = list(nx.topological_sort(G))
        
        for node in topo_order:
            if node in evidence:
                state[node] = evidence[node]
            elif node in self.functions and self.edges[node]:
                # Collect parent values
                parent_vals = [state[p] for p in self.edges[node]]
                state[node] = self.functions[node](*parent_vals)
            else:
                state[node] = self.nodes[node]
                
        return state

    def do_intervention(self, interventions: Dict[str, Any]) -> Dict[str, Any]:
        """
        do-calculus approximation: hard-set a node value, ignoring its parents.
        """
        return self.forward(evidence=interventions)

    def counterfactual(self, base_state: Dict[str, Any], interventions: Dict[str, Any], target_node: str) -> Any:
        """
        Generates a counterfactual: "What would target_node be, if interventions were applied, given base_state?"
        (Simplified implementation assuming exogenous noise is absorbed).
        """
        # In a strict Pearlian SCM, we would first infer exogenous variables (abduction).
        # Here we perform a simplified forward pass overriding with interventions.
        new_evidence = base_state.copy()
        new_evidence.update(interventions)
        result = self.do_intervention(new_evidence)
        return result.get(target_node)
