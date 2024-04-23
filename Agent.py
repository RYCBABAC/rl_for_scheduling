import math
from collections import defaultdict
import random
from sortedcontainers import SortedList
from copy import copy

from State import State
from Global import inf

import torch
from torch import nn, optim
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F


class TabularQLearningAgent:

    # Epsilon property:
    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, given_epsilon):
        self._epsilon = given_epsilon

    # Gamma property:
    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, given_gamma):
        self._gamma = given_gamma

    # Alpha property:
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, given_alpha):
        self._alpha = given_alpha

    @property
    def min_makespan(self):
        return self._min_makespan

    @min_makespan.setter
    def min_makespan(self, given_makespan):
        self._min_makespan = given_makespan

    # Constructor:
    def __init__(self, epsilon, gamma, alpha):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = defaultdict(float)
        self.min_makespan = inf
        self.time = 0
        # creates a counter dict for the states appearances.
        self.counter = defaultdict(int)

    # Methods:
    def getQValue(self, state, action):
        return self.Q[(state, action)]

    def computeValueFromQValues(self, state, time):
        legal_actions = state.get_possible_actions(time, self.min_makespan)
        if len(legal_actions) == 0 and len(state.readyQ) != 0:
            state.readyQ = SortedList()
            for m in state.machines:
                m.job = None
            return - inf

        if len(legal_actions) == 0:
            #     if len(state.readQ) == 0:
            #         return 0
            #     return -self.min_makespan
            #
            return 0

        max_q_value = max([self.getQValue(state, a) for a in legal_actions])
        return max_q_value

    def computeActionFromQValues(self, state, time):
        legal_actions = state.get_possible_actions(time, self.min_makespan)
        if len(legal_actions) == 0:
            return None

        max_q_value = self.computeValueFromQValues(state, time)
        best_actions = []

        for a in legal_actions:
            if self.getQValue(state, a) == max_q_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def flipCoin(self, p):
        r = random.random()
        return r < p

    def get_action(self, state, time):  # Epsilon greedy policy

        if self.alpha != 0:  # we want to count only in training mode
            self.counter[state] += 1

        if self.flipCoin(self.epsilon):  # The random choice
            return random.choice(state.get_possible_actions(time, self.min_makespan))

        # Down here is the greedy choice
        return self.computeActionFromQValues(state, time)

    def update(self, state, action, next_state, reward, time):
        delta = reward + self.gamma * self.computeValueFromQValues(next_state, time) - self.getQValue(state, action)
        if self.alpha < 0:
            self.Q[(state, action)] = self.Q[(state, action)] + -self.alpha * delta
            self.alpha = 1 / (-1 + int(1 / self.alpha))
        else:
            self.Q[(state, action)] = self.Q[(state, action)] + self.alpha * delta

        return None


def get_block(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        #nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )


class Policy(nn.Module):

    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.pi = nn.Sequential(
            get_block(input_size, 128),
            get_block(128, 256),
            nn.Linear(256, output_size),
            nn.Softmax()
        )

        self.V = nn.Sequential(
            get_block(input_size, 128),
            get_block(128, 256),
            nn.Linear(256, 1),
            nn.ReLU()
        )

        self.saved_actions = []
        self.rewards = []

    def forward(self, input_vector):
        action_prob = self.pi(input_vector)  # List of the probabilities for each action
        state_values = self.V(input_vector)  # The value of a state
        return action_prob, state_values



class NNAgent:

    returns = []

    # Epsilon property:
    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, given_epsilon):
        self._epsilon = given_epsilon

    # Gamma property:
    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, given_gamma):
        self._gamma = given_gamma

    # Alpha property:
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, given_alpha):
        self._alpha = given_alpha

    # Constructor:
    def __init__(self, epsilon, gamma, alpha):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.model = Policy(State.n + State.m, State.n + State.m)
        self.optimizer = optim.Adam(self.model.parameters(), self.alpha)
        self.time = 0
        # creates a counter dict for the states appearances.
        self.counter = defaultdict(int)

    # Methods:
    def choose_action(self, state):

        # Gets the possible actions
        actions = state.get_possible_actions()

        # Runs the model on the state to get the probabilities and values
        vector = torch.tensor((state.to_vector())).float()
        probs, state_value = self.model(vector)

        # Gets the wanted action from the distribution
        m = Categorical(probs)
        action_index = m.sample()
        # self.model.saved_actions.append((m.log_prob(action_index), state_value[action_index]))
        self.model.saved_actions.append((m.log_prob(action_index), state_value))

        return actions[action_index % len(actions)]

    def update(self):
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values
        values = []

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, float(R))


        returns = torch.tensor(returns)
        if len(returns.unique()) > 1:
            eps = np.finfo(np.float32).eps.item()
            returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            self.returns.append(float(R))

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            values.append(value)
            # value_losses.append(F.mse_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).mean() + F.mse_loss(torch.tensor(values),torch.tensor(returns))

        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]


