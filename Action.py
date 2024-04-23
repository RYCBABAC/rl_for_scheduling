class Action:

    @property
    def schedule(self):
        return self._schedule

    @schedule.setter
    def schedule(self, other_schedule):
        self._schedule = other_schedule

    def __init__(self, given_schedule):
        self._schedule = None
        self.schedule = given_schedule

    def __eq__(self, other_action):
        if len(self.schedule) != len(other_action.schedule):
            return False

        for i in range(len(self.schedule)):
            if self.schedule[i] != other_action.schedule[i]:
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(self.schedule)