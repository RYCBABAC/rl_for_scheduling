import itertools
import math
import statistics
from copy import copy

from Action import Action
from Global import inf
import LPT
from State import State


class Environment:

    ub = 0
    lb = 0
    # The time property:
    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, current_time):
        self._time = current_time

    # The current state property:
    @property
    def current_state(self):
        return self._current_state

    @current_state.setter
    def current_state(self, state):
        self._current_state = copy(state)

    # The current state property:
    @property
    def initial_state(self):
        return self._initial_state

    @initial_state.setter
    def initial_state(self, state):
        self._initial_state = copy(state)

    # Constructor:
    def __init__(self, initial_state):
        self.time = 0
        self.initial_state = initial_state
        self.current_state = self.initial_state
        self.lb = sum([j.processing_time for j in initial_state.readyQ])/State.m
        self.ub = 0
        # self.inf = sum([j.processing_time for j in self.initial_state.readyQ])

    # Methods:
    def do_action(self, action, makespan):

        if action is None:
            #return copy(self.current_state), -math.inf
            return self.current_state, -inf

        # Adding a job to the corresponding machines
        for j in action.schedule:
            m = self.current_state.machines.pop(0)
            m.job = j
            self.current_state.readyQ.remove(j)
            self.current_state.machines.add(m)

        self.update_machines()
        reward = self.calc_reward(makespan)
        return self.current_state, reward

    def update_machines(self):
        try:
            min_job = next(m.job.processing_time for m in self.current_state.machines if m.job is not None)
        except StopIteration:
            min_job = 0

        self.time += min_job
        for m in reversed(self.current_state.machines):
            if m.job is not None:
                m.job.processing_time -= min_job

                if m.job.processing_time == 0:
                    m.job = None
            else:
                break

    def calc_reward(self, makespan):

        # Option1 - to return minus of the makespan only in the end:
        if not self._current_state.is_final_state():
            return 0
        if self.time > self.ub:
            self.ub = self.time

        return -self.time  # - (self.lb+self.ub) / 2

        # Option2 - to return minus of the variance of all the finish times after each episode:
        # finish_times = [m.finish_time for m in self.current_state.machines]
        # return -statistics.variance(finish_times)

        # # option 3 - to re turn minus of the bound of the makespan
        # machines_amount = len(self.initial_state.machines)
        # machines_times =[m.job.processing_time for m in self.current_state.machines if m.job != None]
        # # machines_time_left = sum(machines_times)
        # # readyQ_time_left = sum([j.processing_time for j in self.current_state.readyQ])
        # readyQ_times = [j.processing_time for j in self.current_state.readyQ]
        # # return - (self.time + (machines_time_left + readyQ_time_left) / machines_amount)
        # # machines_max = max(machines_times) if len(machines_times) != 0 else 0
        # # bound = max(self.time + (machines_time_left + readyQ_time_left) / machines_amount, machines_max)
        # bound = LPT.LPT_schefule(machines_times + readyQ_times, machines_amount)
        # # if bound > makespan: return -inf
        # return -bound



    def reset(self):
        self.current_state = self.initial_state
        self.time = 0
