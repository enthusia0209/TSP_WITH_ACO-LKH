import subprocess
from typing import List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import asyncio
import time
import os

class LKHSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.lkh_path = 'LKH/LKH'  # Update with the actual path to LKH

    def write_tsplib_file(self, distance_matrix: List[List[int]], filename: str, directed):
        """Writes a distance matrix to a TSPLIB formatted file."""
        problem_type = "ATSP" if directed else "TSP"
        n = len(distance_matrix)
        with open(filename, 'w') as f:
            f.write(f"NAME: {problem_type}\nTYPE: {problem_type}\nDIMENSION: {n}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            for row in distance_matrix:
                f.write(" ".join(map(str, row)) + "\n")
            f.write("EOF\n")

    def write_lkh_parameters(self, filename: str, problem_filename: str, tour_filename: str):
        """Writes the parameter file for LKH."""
        with open(filename, 'w') as f:
            f.write(f"PROBLEM_FILE = {problem_filename}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_filename}\n")
            f.write("EOF")

    def run_lkh(self, parameter_file: str):
        """Runs the LKH solver using a given parameter file."""
        result = subprocess.run([self.lkh_path, parameter_file], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"LKH failed: {result.stderr}")
        return result.stdout

    def read_lkh_solution(self, tour_filename: str):
        """Reads the solution produced by LKH."""
        tour = [0]
        with open(tour_filename, 'r') as f:
            lines = f.readlines()
            for line in lines[7:]:
                line = line.strip()
                if line == '-1':
                    break
                tour.append(int(line) - 1)  # Convert 1-based index to 0-based

        tour.append(0)
        return tour

    async def solve(self, distance_matrix, future_id: int, directed=False) -> List[int]:
        is_float = isinstance(distance_matrix[0][0], float)

        # Scale factor for converting float distances to integers if necessary
        scale_factor = 1000 if is_float else 1
        
        scaled_distance_matrix = [
            [int(round(distance * scale_factor)) for distance in row]
            for row in distance_matrix
        ]
        problem_filename = "problem.tsp"
        parameter_filename = "params.par"
        tour_filename = "solution.tour"

        # Write the TSPLIB problem file
        self.write_tsplib_file(scaled_distance_matrix, problem_filename, directed)

        # Write the LKH parameter file
        self.write_lkh_parameters(parameter_filename, problem_filename, tour_filename)

        # Run LKH
        self.run_lkh(parameter_filename)

        # Read and return the solution
        tour = self.read_lkh_solution(tour_filename)

        # Clean up temporary files (optional)
        # os.remove(problem_filename)
        os.remove(parameter_filename)
        os.remove(tour_filename)
        return tour

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == '__main__':
    n_nodes = 100  # Adjust as needed
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = LKHSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    if route is None:
        print(f"{solver.__class__.__name__} No solution found.")
    else:
        print(f"{solver.__class__.__name__} Best Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken: {time.time() - start_time}")