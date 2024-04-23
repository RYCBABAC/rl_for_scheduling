import itertools
import math
from copy import deepcopy
from Action import Action
from Machine import Machine
from Job import Job
from sortedcontainers import SortedList
import numpy as np

class State:

    # Machines property:
    @property
    def machines(self):
        return self._machines

    @machines.setter
    def machines(self, machines_list):
        if type(machines_list) != SortedList:
            raise Exception("Machine list should be a sorted list")

        del self._machines
        self._machines = deepcopy(machines_list)

    # ReadyQ property:
    @property
    def readyQ(self):
        return self._readyQ

    @readyQ.setter
    def readyQ(self, queue):
        if type(queue) != SortedList:
            raise Exception("Ready queue should be a sorted list")

        del self._readyQ
        self._readyQ = deepcopy(queue)

    n = 0
    m = 0

    # Constructor:
    def __init__(self, read_queue, machines_list):
        self._machines = None  # Machine
        self.machines = machines_list  # Machine

        self._readyQ = None  # Jobs
        self.readyQ = read_queue  # Jobs

    # Comparator:
    def __eq__(self, other):
        if len(self.machines) != len(other.machines):
            return False

        if len(self.readyQ) != len(other.readyQ):
            return False

        for (m1, m2) in zip(self.machines, other.machines):
            if m1 != m2:
                return False

        for (q1, q2) in zip(self.readyQ, other.readyQ):
            if q1 != q2:
                return False

        return True

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((tuple(self.machines), tuple(self.readyQ)))

    # Methods:
    def is_initial_state(self):
        return len(self.get_free_machines()) == len(self.machines) and len(self.readyQ) != 0

    def is_final_state(self):
        return len(self.readyQ) == 0 and len(self.get_free_machines()) == len(self.machines)

    def get_free_machines(self):
        free_machines = []
        for m in self.machines:
            if not m.working():
                free_machines.append(m)
            else:
                return free_machines

        if len(free_machines) != 0:
            return free_machines
        raise Exception("There are no free machines something went terribly wrong...")

    def get_possible_actions(self):
        if self.is_initial_state():
            actions = tuple(self.readyQ[-len(self.machines):])
            return [Action(actions)]
        else:
            free = len(self.get_free_machines())
            actions = list(sorted(itertools.combinations(self.readyQ, min(len(self.readyQ), free))))

        return [Action(t) for t in actions]

    # Copy:
    def __copy__(self):
        return State(self.readyQ, self.machines)

    def to_vector(self):
        machine_times = []
        for m in self.machines:
            if m.job is None:
                machine_times.append(0)
            else:
                machine_times.append(m.job.processing_time)

        readyQ_times = [j.processing_time for j in self.readyQ]
        res = np.array(machine_times + readyQ_times)
        size = self.n + self.m
        if len(res) != size:
            res = np.pad(res, (0,size-len(res)))
        return res