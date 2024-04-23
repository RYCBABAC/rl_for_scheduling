import argparse
import math
import random
from statistics import mean

from sortedcontainers import SortedList

import LPT
from Game import Game
from Job import Job
from Machine import Machine
import time

import matplotlib.pyplot as plt

import csv

values = [[5, 3, 1, 20, 3000],
           [90, 30, 1, 200, 3000],
          [150, 30, 1, 200, 4000],
         [60, 30, 200, 500, 3000],
        [90, 30, 200, 500, 3000],
        [150, 30, 200, 500, 4000],
        [60, 40, 1, 200, 3000],
        [90, 40, 1, 200, 3000],
        [150, 40, 1, 200, 4000],
        [60, 40, 200, 500, 3000],
        [90, 40, 200, 500, 3000],
        [150, 40, 200, 500, 4000],
        [60, 50, 1, 200, 3000],
        [90, 50, 1, 200, 3000],
        [150, 50, 1, 200, 4000],
        [60, 50, 200, 500, 3000],
        [90, 50, 200, 500, 3000],
        [150, 50, 200, 500, 4000]]


def get_parameters_from_console():
    parser = argparse.ArgumentParser('')
    parser.add_argument('--sa', dest='sa', action='store_true')
    parser.add_argument('--ra', dest='ra', action='store_true')
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--training', type=int, default=1000)
    parser.add_argument('--testing', type=int, default=10)
    parser.add_argument('--ub', type=int, default=10)
    parser.add_argument('--lb', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--df', type=float, default=1)
    parser.add_argument('--eps', type=float, default=0.2)
    return parser.parse_args()


def jobs_generator(size, lb, ub):
    generated_jobs = SortedList()
    for i in range(size):
        generated_jobs.add(Job(i, random.randint(lb, ub)))
    # generated_jobs.add(Job(0, 2))
    # generated_jobs.add(Job(1, 1))
    # generated_jobs.add(Job(2, 1))
    return generated_jobs


def machines_generator(size):  # at start there are no jobs on the generated machines
    generated_machines = SortedList()
    for i in range(size):
        generated_machines.add(Machine(i, None))
    return generated_machines


def create_random_assigment(jobs_amount, machines_amount, lb, ub, num_of_training, num_of_testing, args):
    # simple case:
    jobs = jobs_generator(jobs_amount, lb, ub)
    machines = machines_generator(machines_amount)
    #print("Jobs: " + str(jobs))
    return Game(jobs, machines, num_of_training, num_of_testing, args), jobs

def new_create_random_assigment(jobs_amount, machines_amount, lb, ub, num_of_training, num_of_testing, args):
    # simple case:
    jobs_times = [2,4,9,9,10,12]
    jobs = SortedList([Job(i,jobs_times[i]) for i in range(len(jobs_times))])
    machines = machines_generator(machines_amount)
    #print("Jobs: " + str(jobs))
    return Game(jobs, machines, num_of_training, num_of_testing, args), jobs

def create_special_assignment(machines_amount, num_of_training, num_of_testing, args):
    jobs = SortedList()
    for i in range(1, 2 * machines_amount):
        jobs.add(Job(i, i))
    machines = machines_generator(machines_amount)
    return Game(jobs, machines, num_of_training, num_of_testing, args)


def run_special_assignment(machines_amount, num_of_training, num_of_testing, args):
    optimal_makespan = 2 * machines_amount - 1

    # The special assignment:
    print(20 * "*" + "run_special_assignment" + 20 * "*")
    print("------------------------------------------------")
    print("Amount of machines: " + str(machines_amount))
    print("Amount of jobs: " + str(2 * machines_amount - 1))
    print("Optimal makespan: " + str(2 * machines_amount - 1))
    print("------------------------------------------------")
    print(" Number of training episodes: " + str(num_of_training))
    print(" Number of testing episodes: " + str(num_of_testing))
    print("------------------------------------------------")

    assignment = create_special_assignment(machines_amount, num_of_training, num_of_testing, args)
    makespans = assignment.run()

    print("The test makespans list: " + str(makespans) + " -> average makespan: " + str(mean(makespans)))

    success_amount = sum([1 for m in makespans if m == optimal_makespan])
    success_rate = success_amount / len(makespans)
    print("success amount = " + str(success_amount) + ", success rate = " + str(100 * success_rate) + "%")
    print("------------------------------------------------")


def run_random_assignment_special_sizes(machines_amount, num_of_training, num_of_testing, args):
    # The random assignment (special sizes):
    print(20 * "*" + "random_assignment_special_sizes" + 20 * "*")
    print("------------------------------------------------")
    print("Amount of machines: " + str(machines_amount))
    print("Amount of jobs: " + str(2 * machines_amount - 1))
    print("------------------------------------------------")
    print(" Number of training episodes: " + str(num_of_training))
    print(" Number of testing episodes: " + str(num_of_testing))
    print("------------------------------------------------")

    rng = 2 * machines_amount - 1
    assignment = create_random_assigment(2 * machines_amount - 1, rng, machines_amount, num_of_training, num_of_testing,
                                         args)
    print("------------------------------------------------")
    makespans = assignment.run()
    print("------------------------------------------------")

    ideal_schedul = sum([j.processing_time for j in assignment.env.initial_state.readyQ]) / machines_amount
    print("Bound of an ideal schedule makespan: " + str(ideal_schedul))

    print("------------------------------------------------")


def run_random_assignment(machines_amount, num_of_training, num_of_testing, args):
    game = create_random_assigment(args.n, args.ub, args.lb, machines_amount, num_of_training, num_of_testing, args)
    start = time.time()
    game.run()
    end = time.time()
    print("Time took to run " + str(end - start) + " seconds")
    LPT_time = LPT.LPT_schefule(game.env.initial_state.readyQ, len(game.env.initial_state.machines))
    print(str(LPT_time))


def jobs_to_str(jobs):
    res = "{"
    for j in jobs:
        res += str(j.processing_time) + ","
    res = res[:-1] + "}"
    return res

def calc_bound(jobs,m):
    return max(math.ceil(mean([j.processing_time for j in jobs])),jobs[-1].processing_time, jobs[-m].processing_time+jobs[-m-1].processing_time)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    args = get_parameters_from_console()
    #
    # m = args.m
    # training = args.training
    # testing = args.testing
    #
    # run_function = run_random_assignment
    # if args.sa and not args.ra:
    #     run_function = run_special_assignment
    # elif args.sa and args.ra:
    #     run_function = run_random_assignment_special_sizes
    # run_function(m, training, testing, args)

    iter_num = 10
    res = []
    R = []
    for row in values:
        n = row[0]
        m = row[1]
        lb = row[2]
        ub = row[3]
        training = row[4]
        for i in range(0, iter_num):
            game, jobs = new_create_random_assigment(n, m, lb, ub, training, 10, args)
            LPT_makespan = LPT.LPT_schefule(game.env.initial_state.readyQ, len(game.env.initial_state.machines))
            game.agent.min_makespan = LPT_makespan
            game.env.inf = LPT_makespan
            bound = calc_bound(game.env.initial_state.readyQ, m)
            start = time.time()
            makespans, once, five_times,returns = game.run()
            end = time.time()
            iter_res = [n, m, lb, ub, training, jobs_to_str(jobs), LPT_makespan, mean(makespans), end - start, once, five_times, bound]
            R.append(returns)
            res.append(iter_res)
            plt.plot(returns)
        break
    file_closed = False
    while not file_closed:
        try:
            f = open("data.csv", 'w+', newline='')
            file_closed = True
        except:
            input("File is open, close it and press enter...> ")

    writer = csv.writer(f)
    writer.writerow(["n", "m", "lb", "ub", "training", "jobs", "LPT", "RL", "time", "once", "five times", "bound"])
    writer.writerows(res)
    f.close()

    f = open("returns.csv", 'w+', newline='')
    file_closed = True
    writer = csv.writer(f)
    writer.writerow(R[0])
    f.close()

