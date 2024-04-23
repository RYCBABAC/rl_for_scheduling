from copy import copy

from Job import Job


class Machine:

    # The id property:
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, machine_id):
        self._id = machine_id

    # The finish_time property:
    @property
    def finish_time(self):
        return self._finish_time

    @finish_time.setter
    def finish_time(self, time):
        self._finish_time = time

    # The job property:
    @property
    def job(self):
        return self._job

    @job.setter
    def job(self, other_job):
        if other_job is not None and type(other_job) != Job:
            raise Exception("Invalid input")
        self._job = copy(other_job)
        if self._job is not None:
            self.finish_time += self._job.processing_time

    # Constructors:
    def __init__(self, machine_id, job):
        self.id = machine_id
        self.finish_time = 0
        self.job = job


    # Comparator:
    def __eq__(self, other):
        return self.job == other.job

    def __lt__(self, other_machine):
        if self.job is None:
            return True
        if other_machine.job is None:
            return False
        return self.job < other_machine.job

    def __ne__(self, other):
        return not(self == other)

    # To string
    def __str__(self):
        return "MachineID: " + str(self.id) + "\tJob: " + str(self.job) + "\n"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        # return hash((self.id, self.job))
        return hash(self.job)


    # Methods:
    def calc_finish_time(self, total_working_time):
        if self.job is not None:
            return total_working_time + self.job.processing_time
        return -1

    def working(self):
        return self.job is not None

    # Copy:
    def __copy__(self):
        return Machine(self.id, self.job)
