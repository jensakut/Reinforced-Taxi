import numpy as np
from collections import defaultdict
import random 

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.i_episode = 0
        self.eps = 1.0
        self.eps_min = 0.0
        
        # parameters 
        self.alpha = 0.1

        self.gamma = 1.0
        
        
        
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        if random.random() > self.eps: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return random.choice(np.arange(self.nA))


    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Use sarsa_maexp
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        # sarsamax 
        
        current = self.Q[state][action]         # estimate in Q-table (for current state, action pair)
        policy_s = np.ones(self.nA) * self.eps / self.nA  # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA) # greedy action
        Qsa_next = np.dot(self.Q[next_state], policy_s)         # get value of state at next time step
        target = reward + (self.gamma * Qsa_next)               # construct target
        self.Q[state][action] = current + (self.alpha * (target - current)) # get updated value 
        
        
        """Sarsa-Max / Q-Learning: Returns updated Q-value for the most recent experience.
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0  # value of next state 
        target = reward + (self.gamma * Qsa_next)               # construct TD target
        self.Q[state][action] = current + (self.alpha * (target - current)) # get updated value 
        """
        if done: 
            self.i_episode += 1
            self.eps = max(1.0 / self.i_episode, self.eps_min)

