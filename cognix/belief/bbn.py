"""
Bayesian Belief Networks (BBN) exact inference for Directed Acyclic Graphs (DAGs).
"""
import numpy as np
from typing import Dict

class BayesianBeliefNetwork:
    """
    Implements a discrete BBN over agents.
    """
    def __init__(self):
        self.nodes = []
        self.edges = {}
        self.cpts = {} # Conditional Probability Tables

    def add_node(self, name: str, cpt: np.ndarray):
        """
        Adds a node with its Conditional Probability Table.
        For root nodes, CPT is 1D array of prior probabilities.
        For child nodes, CPT is multi-dimensional array mapping parent states to child states.
        """
        if name not in self.nodes:
            self.nodes.append(name)
            self.edges[name] = []
            self.cpts[name] = cpt

    def add_edge(self, parent: str, child: str):
        """Adds a directed edge."""
        if child not in self.edges:
            self.edges[child] = []
        self.edges[child].append(parent)

    def exact_inference(self, evidence: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Naive exact inference using full joint distribution (only viable for small graphs).
        """
        # A true production BBN would use Variable Elimination or Junction Trees.
        # This is a conceptual stub for the BBN integration.
        return {}
