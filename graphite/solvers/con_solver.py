import subprocess
from typing import List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import asyncio
import time
import os

class CONSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)
        self.concorde_path = 'concorde/TSP/concorde'  # Update with the actual path to Concorde

    def write_tsplib_file(self, distance_matrix: List[List[int]], filename: str, directed):
        """Writes a distance matrix to a TSPLIB formatted file."""
        problem_type = "ATSP" if directed else "TSP"
        n = len(distance_matrix)
        with open(filename, 'w') as f:
            f.write(f"NAME: {problem_type}\nTYPE: {problem_type}\nDIMENSION: {n}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            for index, row in enumerate(distance_matrix, start=1):
                f.write(f"" + " ".join(map(str, row)) + "\n")
            f.write("EOF\n")

    def run_concorde(self, tsplib_filename: str, tour_filename: str):
        """Runs the Concorde solver using a TSPLIB file."""
        result = subprocess.run([self.concorde_path, '-o', tour_filename, tsplib_filename], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Concorde failed: {result.stderr}")
        return result.stdout

    def read_concorde_solution(self, tour_filename: str):
        """Reads the solution produced by Concorde."""
        tour = [0]
        with open(tour_filename, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.strip()
            #     if line == '-1':
            #         break
            #     tour.append(int(line) - 1)  # Convert 1-based index to 0-based

        tour.append(0)  # Close the tour
        return tour

    async def solve(self, distance_matrix, future_id: int, directed=False) -> List[int]:
        problem_filename = "problem.tsp"
        tour_filename = "solution.sol"

        # Write the TSPLIB problem file
        self.write_tsplib_file(distance_matrix, problem_filename, directed)

        # Run Concorde
        self.run_concorde(problem_filename, tour_filename)

        # Read and return the solution
        tour = self.read_concorde_solution(tour_filename)

        # Clean up temporary files (optional)
        # os.remove(problem_filename)
        os.remove(tour_filename)
        return tour

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == '__main__':
    n_nodes = 100  # Adjust as needed
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = CONSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    if route is None:
        print(f"{solver.__class__.__name__} No solution found.")
    else:
        print(f"{solver.__class__.__name__} Best Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken: {time.time() - start_time}")
