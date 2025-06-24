import numpy as np
import random
import math
import time
import sys

'''
Sparsify a given matrix
'''

class RowCount:
    def __init__(self, dirty_bit=0, ones_count=0, i=0):
        self.dirty_bit = dirty_bit
        self.ones_count = ones_count
        self.i = i


def random_index(M):
    return random.randint(0, M - 1)


def count_ones(N, M, S):
    return sum(S[i][j] for i in range(M) for j in range(N))


def write_matrix(file_out_name, N, M, S):
    with open(file_out_name, "w") as f:
        f.write(f"{count_ones(N, M, S)}\n")
        f.write(f"{M} {N}\n")
        for i in range(M):
            f.write(" ".join(str(S[i][j]) for j in range(N)) + "\n")


def check_order(A, M):
    # Assuming A is already sorted, you can remove the assert in production
    for i in range(1, M):
        assert A[i - 1].dirty_bit <= A[i].dirty_bit


def print_A(A, M):
    for i in range(M):
        print(f"[D: {A[i].dirty_bit:2} O: {A[i].ones_count:2} i: {A[i].i:2}]  ", end=" ")
    print()


def print_matrix(N, M, S):
    for i in range(M):
        print(" ".join(str(S[i][j]) for j in range(N)))


def swap_structs(a, b):
    a.dirty_bit, b.dirty_bit = b.dirty_bit, a.dirty_bit
    a.ones_count, b.ones_count = b.ones_count, a.ones_count
    a.i, b.i = b.i, a.i


def i_swap(a, b):
    a[0], b[0] = b[0], a[0]


def simulated_annealing(argv):
    # Handle command line arguments and matrix initialization
    if len(argv) < 3 or len(argv) > 9:
        print_usage()
        return

    input_file = argv[1]
    output_file = argv[2]
    steps = int(argv[3]) if len(argv) > 3 else 100
    iter_count = int(argv[4]) if len(argv) > 4 else 100
    hot_hit_perc = float(argv[5]) if len(argv) > 5 else 0.05
    hot_hit_prob = float(argv[6]) if len(argv) > 6 else 0.05
    cold_hit_perc = float(argv[7]) if len(argv) > 7 else 0.05
    cold_hit_prob = float(argv[8]) if len(argv) > 8 else 0.01

    # File I/O and matrix setup
    with open(input_file, "r") if input_file != "stdin" else sys.stdin as f:
        N = int(f.readline().strip())  # Number of columns
        M = int(f.readline().strip())  # Number of rows

        max_col_size = int(f.readline().strip())
        max_row_size = int(f.readline().strip())

        # Skip column and row info
        for _ in range(N):
            f.readline()
        for _ in range(M):
            f.readline()

        # Matrix initialization
        S = [[0] * N for _ in range(M)]
        cost = 0
        for j in range(N):
            for _ in range(max_col_size):
                t = int(f.readline().strip())
                if t > 0:
                    cost += 1
                    S[t - 1][j] = 1

        A = [RowCount() for _ in range(M)]
        P = list(range(M))

        for i in range(M):
            A[i].ones_count = sum(S[i])
            A[i].i = i

        dirties = M
        new_row = [0] * N

        # Annealing parameters
        temp = 0.05 * N
        sc = hot_hit_prob
        temp /= -math.log(sc)

        cold = 0.05 * N
        sc = cold_hit_prob
        cold /= -math.log(sc)

        alpha = 1.0 / steps
        alpha = (cold / temp) ** alpha

        # Initialize variables for simulation
        min_cost = cost
        pmin = min_cost + 1
        total_ops = 0
        total_iterations = 0
        clock_begin = time.time()

        print("# Cost MinCost Total_Time Total_Ops Total_Iterations Dirty Temperature")

        while pmin > min_cost or temp > cold:
            pmin = min_cost
            print(
                f"{cost:10} {min_cost:10} {time.time() - clock_begin:e} {total_ops:20} {total_iterations:20} {dirties:6} {temp:e}")

            k = 1
            while k < iter_count:
                delta_cost = 0
                if dirties > 0:
                    i = random_index(dirties)
                    ii = 0
                    while ii < M and delta_cost >= 0:
                        i_swap(P[ii], P[ii + random.randint(0, M - ii - 1)])
                        if P[ii] != i:
                            delta_cost = 0
                            for j in range(N):
                                total_ops += 1
                                new_row[j] = (S[A[i].i][j] + S[A[P[ii]].i][j]) % 2
                                delta_cost += new_row[j]

                            if A[i].ones_count > A[P[ii]].ones_count:
                                delta_cost -= A[i].ones_count
                            else:
                                delta_cost -= A[P[ii]].ones_count

                            if delta_cost < 0:
                                if A[i].ones_count > A[P[ii]].ones_count:
                                    ip = i
                                    i = P[ii]
                                else:
                                    ip = P[ii]
                            ii += 1

                    if delta_cost >= 0:
                        A[i].dirty_bit = 1
                        dirties -= 1
                        swap_structs(A[i], A[dirties])

                if delta_cost >= 0:
                    i = random_index(M)
                    ip = random_index(M - 1)
                    if ip >= i:
                        ip += 1

                    delta_cost = 0
                    for j in range(N):
                        total_ops += 1
                        new_row[j] = (S[A[i].i][j] + S[A[ip].i][j]) % 2
                        delta_cost += new_row[j]
                    delta_cost -= A[ip].ones_count

                if delta_cost <= 0 or (random.random() < math.exp(-delta_cost / temp)):
                    for j in range(N):
                        S[A[ip].i][j] = new_row[j]
                    A[ip].ones_count += delta_cost
                    cost += delta_cost

                    if A[ip].dirty_bit == 1:
                        A[ip].dirty_bit = 0
                        swap_structs(A[ip], A[dirties])
                        dirties += 1

                    if cost < min_cost:
                        min_cost = cost

                k += 1
                total_iterations += 1

            temp *= alpha

        write_matrix(output_file, N, M, S)
        print("\nFinished!")


def print_usage():
    print("SYNOPSIS")
    print("./project <input.alist> <output.txt> [<steps> [<iter> [<hotPer> [<hotPr> [<coldPer> [<coldPr>]]]]]]")
    print("\nDESCRIPTION")
    print("\n The <input.alist> is the input parity check matrix in alist format. If")
    print(" this string is exactly stdin then standard input is used instead. The")
    print(" <output.txt> is the name of the file to store the output matrix.\n")


'''
# argv arguements passed 
./project <input.alist> <output.txt> [<steps> [<iter> [<hotPer> [<hotPr> [<coldPer> [<coldPr>]]]]]]

    input_file = argv[1]
    output_file = argv[2]
    steps = int(argv[3]) if len(argv) > 3 else 100
    iter_count = int(argv[4]) if len(argv) > 4 else 100
    hot_hit_perc = float(argv[5]) if len(argv) > 5 else 0.05
    hot_hit_prob = float(argv[6]) if len(argv) > 6 else 0.05
    cold_hit_perc = float(argv[7]) if len(argv) > 7 else 0.05
    cold_hit_prob = float(argv[8]) if len(argv) > 8 else 0.01
'''
if __name__ == "__main__":
    simulated_annealing(sys.argv)


















