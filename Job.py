class Job:

    # Processing time property:
    @property
    def processing_time(self):
        return self._processing_time

    @processing_time.setter
    def processing_time(self, time):
        self._processing_time = time

    # Id property:
    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, job_id):
        self._id = job_id

    # Constructor:
    def __init__(self, job_id, processing_time):
        self.id = job_id
        self.processing_time = processing_time  # Will be changed

    # Comparators:
    def __eq__(self, other_job):
        if other_job is None: return False
        return other_job.processing_time == self.processing_time

    def __lt__(self, other):
        return self.processing_time < other.processing_time

    def __ne__(self, other_job):
        return not (self == other_job)

    def __int__(self):
        return self.processing_time

    # To string:
    def __str__(self):
        # return "JobID: " + str(self.id) + ", Processing time: " + str(self.processing_time) + "\n"
        return "J" + str(self.id) + "(-" + str(self.processing_time) + "-)"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        # return hash((self.id, self.processing_time))
        return hash(self.processing_time)

    # Copy:
    def __copy__(self):
        return Job(self.id, self.processing_time)

    def __add__(self, other):
        return int(self) + int(other)