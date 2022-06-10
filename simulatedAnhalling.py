import math
from math import exp
import time
import random
from copy import copy


def create_board(n):
    board = list(range(n))

    random.shuffle(board)

    return board


def display(board):
    size = len(board)
    if size <= 40:
        print('Displaying board of size', size)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[j] == i:
                    print('1 ', end="")
                else:
                    print('0 ', end="")
            print()


def EvalutionFunction(board):
    qh = 0
    count = 0;
    found = []
    for qv in board:
        c1 = (qv, qh)
        q1h = 0
        # print(c1,end='->')
        for q1v in board:
            c2 = (q1v, q1h)
            q1h = q1h + 1
            if c1 != c2:
                if ((c1, c2) not in found) and ((c2, c1) not in found):
                    if c1[0] == c2[0]:
                        count = count + 1;
                        found.append((c1, c2))

                    if c1[1] == c2[1]:
                        count = count + 1;
                        found.append((c1, c2))

                    if c1[0] - c2[0] == c1[1] - c2[1]:
                        count = count + 1;
                        found.append((c1, c2))

                    if c1[0] - c2[0] == c2[1] - c1[1]:
                        count = count + 1;
                        found.append((c1, c2))
        qh = qh + 1
    return count


def simulated_annealing(current_solution, temp, cool_down_tax):
    queens = len(current_solution)  # pegando tamanho do tabuleiro
    iterations = 1
    reached_global_max = False

    current_solution_cost = EvalutionFunction(current_solution)

    print("Initial solution with", current_solution_cost, "conflicts")
    print("Calculating best solution...\n")

    while temp > 1 / 1000 and not reached_global_max:
        if iterations % 1000 == 0:
            print("Iteration:", iterations, " - Temperature:", temp)

        iterations += 1
        temp *= cool_down_tax

        NewqueenPos = [random.randrange(0, queens), random.randrange(0, queens)]
        while NewqueenPos[0] == NewqueenPos[1]:
            NewqueenPos = [random.randrange(0, queens), random.randrange(0, queens)]

        newSolution = copy(current_solution)

        tempr = newSolution[NewqueenPos[0]]
        # print(aux)
        newSolution[NewqueenPos[0]] = newSolution[NewqueenPos[1]]
        newSolution[NewqueenPos[1]] = tempr

        new_cost = EvalutionFunction(newSolution)

        variation = new_cost - current_solution_cost

        if variation < 0 or random.uniform(0, 1) < exp(-variation / temp):
            current_solution = copy(newSolution)
            current_solution_cost = new_cost

        if current_solution_cost == 0:
            reached_global_max = True

    return iterations, current_solution


if __name__ == "__main__":
    start = time.time()
    value = input("Enter No of Queens: ")
    no_of_queens = int(value)
    temp = 1
    cool_down = 1 - (1 / 1000)

    initial_solution = create_board(no_of_queens)
    print(initial_solution)
    display(initial_solution)
    iterations, solution = simulated_annealing(
        initial_solution,
        temp,
        cool_down)

    display(solution)

    print("\nBest solution found with", EvalutionFunction(solution), "conflict(s)")
    print("Number of Iterations: ", iterations)
    print("Done in :", time.time() - start, " seconds.")
