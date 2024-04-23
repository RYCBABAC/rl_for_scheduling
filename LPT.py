def get_free_machines_indices(machines_working_time):
    return [i for i in range(len(machines_working_time)) if machines_working_time[i] == min(machines_working_time)]


def LPT_schefule(readyQ, machines_amount):
    sorted_readyQ = sorted(readyQ)
    machines_working_times = [0 for i in range(machines_amount)]

    while len(sorted_readyQ) != 0:
        free_machines_indices = get_free_machines_indices(machines_working_times)
        for i in free_machines_indices:
            machines_working_times[i] += int(sorted_readyQ.pop())
            if len(sorted_readyQ) == 0:
                break

    return max(machines_working_times)