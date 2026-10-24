"""
PettingZoo Agent-Environment Cycle (AEC) Wrapper for COGNIX.
"""
try:
    from pettingzoo import AECEnv
    from pettingzoo.utils import agent_selector
    import gymnasium as gym
    import numpy as np
except ImportError:
    agent_selector = None
    AECEnv = object
    import numpy as np

class CognixAECEnv:
    """
    Toy Cooperative Navigation Environment following PettingZoo AEC API.
    Used for testing MADDPG/QMIX coordination.
    """
    metadata = {"render_modes": ["human"], "name": "cognix_nav_v0"}

    def __init__(self, num_agents: int = 3):
        if agent_selector is None:
            raise ImportError(
                "CognixAECEnv requires 'pettingzoo' and 'gymnasium'. "
                "Install with: pip install 'cognix-uncertainty-framework[marl]'"
            )
        self.num_agents = num_agents
        self.possible_agents = [f"agent_{i}" for i in range(num_agents)]
        self.agents = self.possible_agents[:]
        
        # Action space: Discrete (e.g., UP, DOWN, LEFT, RIGHT, STAY)
        self.action_spaces = {agent: gym.spaces.Discrete(5) for agent in self.possible_agents}
        # Observation space: (x, y) self pos, (x, y) target pos
        self.observation_spaces = {agent: gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32) for agent in self.possible_agents}
        
        self.rewards = {agent: 0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = {agent: {} for agent in self.possible_agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.agent_selection = self._agent_selector.reset()
        self.state = {agent: np.random.rand(4) for agent in self.agents}
        
    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        
        # Toy transition logic (simulate movement)
        self.state[agent][:2] += np.random.randn(2) * 0.1
        self.state[agent][:2] = np.clip(self.state[agent][:2], 0, 1)
        
        # Reward: negative distance to target
        dist = np.linalg.norm(self.state[agent][:2] - self.state[agent][2:])
        self.rewards[agent] = -float(dist)
        
        if dist < 0.1:
            self.terminations[agent] = True
            self.rewards[agent] += 10.0
            
        if self._agent_selector.is_last():
            for a in self.agents:
                self.truncations[a] = False # Add truncation logic if needed
                
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> np.ndarray:
        return self.state[agent]
        
    def _was_dead_step(self, action):
        pass # PettingZoo compliance
