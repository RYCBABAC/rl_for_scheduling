import math
import random
from collections import defaultdict
from copy import copy, deepcopy
from sortedcontainers import SortedList

import Global
from Environment import Environment
from Job import Job
from Machine import Machine
from State import State
from Agent import TabularQLearningAgent
from Agent import NNAgent


class Game:

    @property
    def env(self):
        return self._env

    @env.setter
    def env(self, environment):
        self._env = environment

    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, learning_agent):
        self._agent = learning_agent

    @property
    def num_of_training(self):
        return self._num_of_training

    @num_of_training.setter
    def num_of_training(self, num):
        self._num_of_training = num

    @property
    def num_of_testing(self):
        return self._num_of_testing

    @num_of_testing.setter
    def num_of_testing(self, num):
        self._num_of_testing = num

    def __init__(self, jobs, machines, num_of_training, num_of_testing, args):
        State.n = len(jobs)
        State.m = len(machines)
        init_state = State(jobs, machines)
        self.env = Environment(init_state)
        # self.agent = TabularQLearningAgent(args.eps, args.df, args.lr, sum([j.processing_time for j in self.env.initial_state.readyQ]))  # epsilon, gamma, alpha
        self.agent = NNAgent(args.eps, args.df, args.lr)  # epsilon, gamma, alpha
        self.num_of_training = num_of_training
        self.num_of_testing = num_of_testing
        Global.inf = sum([j.processing_time for j in jobs])

    # Game methods:\
    def operate_schedule(self):
        reward = 0
        while not self.env.current_state.is_final_state():
            env_state = deepcopy(self.env.current_state)
            agent_action = self.agent.choose_action(env_state)
            next_env_state, reward = self.env.do_action(agent_action, self.agent.min_makespan)
            self.agent.model.rewards.append(reward)
        self.agent.update()

        return self.env.time, reward

    def operate_schedule_with_prints(self):
        printer = self.initial_state_print()
        reward = 0
        # print("time: " + str(self.env.time))
        while not self.env.current_state.is_final_state():
            # self.show_current_state(printer)
            env_state = deepcopy(self.env.current_state)
            # for a in env_state.get_possible_actions(): print(a.schedule)

            agent_action = self.agent.get_action(env_state)
            self.update_printer(printer, agent_action)
            # print(agent_action.schedule)

            next_env_state, reward = self.env.do_action(agent_action)
            self.agent.update(env_state, agent_action, next_env_state, reward)
            # print("time: " + str(self.env.time))

        # self.env.current_state.finish_schedule()
        # self.show_current_state(printer)
        # print("time: " + str(self.env.time))
        print("operate_schedule_with_prints - final reward: " + str(1 / reward))
        return (self.env.time, reward)

    def run(self):

        # Training mode
        rewards = []
        for i in range(self.num_of_training):
            self.env.reset()
            self.operate_schedule()

        # Testing mode
        self.agent.alpha = 0
        self.agent.epsilon = 0
        makespans = []

        for i in range(self.num_of_testing):
            self.env.reset()
            # self.operate_schedule()
            makespan, current_reward = self.operate_schedule()
            makespans.append(makespan)

        #print("------------------------------------------------")
        # once, five_times = self.show_counter()
        # self.env.reset()
        # self.operate_schedule_with_prints()
        once = 0
        five_times = 0
        return makespans, once, five_times, self.agent.returns

    # Printing methods:
    def initial_state_print(self):
        printer = [""] * len(self.env.initial_state.machines)
        for (i, m) in enumerate(self.env.initial_state.machines):
            printer[i] = "M" + str(m.id) + ": "
        return printer

    def update_printer(self, printer, action):
        for (index, job) in enumerate(action.schedule):
            printer[index] += str(job) + ", "

    def show_current_state(self, printer):
        for status in printer:
            print(status)

    def show_counter(self):
        # using the counter dict we have created
        counter = self.agent.counter
        one_appearance = [state for state, count in counter.items() if count == 1]
        less_then_five_appearances = [state for state, count in counter.items() if count < 5]

        # the number of visited states
        num_of_states = len(counter)

        # prints to screen the results
        # print("What percentage of the visited states were visited only once? ",
        #       100 * len(one_appearance) / num_of_states, "%")
        # print("What percentage of the visited states were visited less than five times? ",
        #       100 * len(less_then_five_appearances) / num_of_states, "%")
        return 100 * len(one_appearance) / num_of_states, 100 * len(less_then_five_appearances) / num_of_states