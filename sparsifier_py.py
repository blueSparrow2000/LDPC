import argparse
import numpy as np
import random
import math
import time

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Simulated Annealing to sparsify a matrix')

    # Required arguments
    parser.add_argument('input_file', type=str, help='Input matrix file in alist format or "stdin"')
    parser.add_argument('output_file', type=str, help='Output file to store result matrix')

    # Optional arguments with defaults
    parser.add_argument('--steps', type=int, default=100, help='Number of temperatures between hot and cold temperatures')
    parser.add_argument('--iter', type=int, default=100, help='Number of iterations at each temperature')
    parser.add_argument('--hot_percentage', type=float, default=0.05, help='Hit percentage for hot temperature')
    parser.add_argument('--hot_probability', type=float, default=0.05, help='Probability of accepting hot temperature')
    parser.add_argument('--cold_percentage', type=float, default=0.05, help='Hit percentage for cold temperature')
    parser.add_argument('--cold_probability', type=float, default=0.01, help='Probability of accepting cold temperature')

    return parser.parse_args()

# Function to read matrix in alist format
def read_matrix(input_file):
    if input_file == "stdin":
        return np.array([list(map(int, line.strip().split())) for line in sys.stdin.readlines()])
    else:
        with open(input_file, 'r') as f:
            return np.array([list(map(int, line.strip().split())) for line in f.readlines()])

# Function to write matrix to file
def write_matrix(output_file, matrix):
    with open(output_file, 'w') as f:
        f.write(f"{np.count_nonzero(matrix)}\n")  # Count of non-zero entries
        f.write(f"{matrix.shape[0]} {matrix.shape[1]}\n")
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')

# Simulated annealing function to sparsify the matrix
def simulated_annealing(matrix, steps, iterations, hot_perc, hot_prob, cold_perc, cold_prob):
    N, M = matrix.shape
    cost = np.count_nonzero(matrix)  # Initial number of ones
    temp = hot_perc * N  # Initial temperature
    cold = cold_perc * N  # Cold temperature
    alpha = pow((cold / temp), (1 / steps))  # Temperature decay rate

    A = [{'dirty': 0, 'ones_count': np.sum(matrix[i]), 'i': i} for i in range(M)]
    dirties = M

    total_ops = 0
    total_iterations = 0

    start_time = time.time()

    print(f"# Cost\tMinCost\tTotal_Time\tTotal_Ops\tTotal_Iterations\tDirty\tTemperature")

    while temp > cold:
        min_cost = cost
        prev_min_cost = min_cost + 1

        # Display the current status
        print(f"{cost}\t{min_cost}\t{(time.time() - start_time):.6f}\t{total_ops}\t{total_iterations}\t{dirties}\t{temp:.6f}")

        for _ in range(iterations):
            delta_cost = 0
            if dirties > 0:
                i = random.choice(range(dirties))
                j = random.choice([x for x in range(M) if x != i])  # Select a second row randomly

                # Calculate delta cost by flipping rows i and j
                new_row = (matrix[A[i]['i']] + matrix[A[j]['i']]) % 2
                delta_cost = np.sum(new_row) - A[i]['ones_count'] - A[j]['ones_count']

                if delta_cost <= 0 or random.random() < math.exp(-delta_cost / temp):
                    matrix[A[i]['i']] = new_row  # Apply the change
                    A[i]['ones_count'] += delta_cost
                    cost += delta_cost
                    total_ops += N  # Each row update involves N operations

                    if A[i]['dirty'] == 1:
                        A[i]['dirty'] = 0
                        A[dirties-1], A[i] = A[i], A[dirties-1]
                        dirties += 1

            total_iterations += 1

        temp *= alpha  # Decrease the temperature

    return matrix

# Main function to run the program
def main():
    # Parse arguments
    args = parse_args()

    # Read the input matrix
    matrix = read_matrix(args.input_file)

    # Perform simulated annealing
    result_matrix = simulated_annealing(
        matrix,
        steps=args.steps,
        iterations=args.iter,
        hot_perc=args.hot_percentage,
        hot_prob=args.hot_probability,
        cold_perc=args.cold_percentage,
        cold_prob=args.cold_probability
    )

    # Write the resulting matrix to output file
    write_matrix(args.output_file, result_matrix)

if __name__ == "__main__":
    main()
