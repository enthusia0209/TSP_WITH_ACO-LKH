import subprocess
from typing import List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import asyncio
import time
import os
import random

class AntColonyTSP:
    def __init__(self, num_ants=50, alpha=1, beta=5, evaporation_rate=0.6, pheromone_init=0.01):
        self.num_ants = num_ants
        self.alpha = alpha  # Pheromone influence
        self.beta = beta    # Distance influence
        self.evaporation_rate = evaporation_rate
        self.pheromone_init = pheromone_init

    def initialize_pheromones(self, num_cities, initial_tour=None):
        """Initialize pheromone values. Boost pheromones for edges in the initial tour."""
        pheromones = [[self.pheromone_init for _ in range(num_cities)] for _ in range(num_cities)]

        if initial_tour:
            # Increase pheromone levels for edges in the LKH tour
            for i in range(len(initial_tour) - 1):
                city1 = initial_tour[i]
                city2 = initial_tour[i + 1]
                pheromones[city1][city2] += 0.5  # Increase pheromone level
                pheromones[city2][city1] += 0.5  # Ensure symmetry

        return pheromones

    def ant_solution(self, pheromones, distance_matrix):
        """Construct a solution by simulating an ant's tour starting from city 1 (index 0)."""
        num_cities = len(distance_matrix)
        unvisited = set(range(1, num_cities))  # Ant starts from city 1 (index 0)
        tour = [0]  # Start from city 1 (index 0)

        current_city = 0  # City 1 is represented by index 0
        while unvisited:
            next_city = self.choose_next_city(current_city, unvisited, pheromones, distance_matrix)
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        tour.append(0)  # Return to city 1 (index 0) to complete the tour
        return tour

    def choose_next_city(self, current_city, unvisited, pheromones, distance_matrix):
        """Choose the next city based on pheromones and distance heuristics."""
        probabilities = []
        total_pheromone = 0.0

        for city in unvisited:
            pheromone = pheromones[current_city][city] ** self.alpha
            distance = (1.0 / distance_matrix[current_city][city]) ** self.beta
            probability = pheromone * distance
            probabilities.append(probability)
            total_pheromone += probability

        if total_pheromone == 0:
            return random.choice(list(unvisited))

        probabilities = [p / total_pheromone for p in probabilities]
        rand = random.random()
        cumulative_probability = 0.0
        for i, city in enumerate(unvisited):
            cumulative_probability += probabilities[i]
            if rand < cumulative_probability:
                return city

    def update_pheromones(self, pheromones, tours, distance_matrix):
        """Evaporate pheromones and reinforce them based on the tours."""
        num_cities = len(pheromones)

        for i in range(num_cities):
            for j in range(num_cities):
                pheromones[i][j] *= (1 - self.evaporation_rate)

        for tour in tours:
            tour_length = self.get_tour_distance(tour, distance_matrix)
            pheromone_deposit = 1.0 / tour_length  # The better the tour, the more pheromone deposited
            for i in range(len(tour) - 1):
                pheromones[tour[i]][tour[i + 1]] += pheromone_deposit
                pheromones[tour[i + 1]][tour[i]] += pheromone_deposit  # For undirected graphs

    def get_tour_distance(self, tour, distance_matrix):
        """Calculate the total distance of a given tour."""
        return sum(distance_matrix[tour[i]][tour[i+1]] for i in range(len(tour) - 1))

    def solve(self, distance_matrix, initial_tour):
        """Solve the TSP using ACO, starting with the LKH tour as a guide."""
        num_cities = len(distance_matrix)
        pheromones = self.initialize_pheromones(num_cities, initial_tour=initial_tour)

        best_tour = None
        best_length = float('inf')

        for iteration in range(5):  # Number of iterations
            tours = []
            for ant in range(self.num_ants):
                tour = self.ant_solution(pheromones, distance_matrix)
                tours.append(tour)

                tour_length = self.get_tour_distance(tour, distance_matrix)
                if tour_length < best_length:
                    best_length = tour_length
                    best_tour = tour

            self.update_pheromones(pheromones, tours, distance_matrix)

        return best_tour, best_length


class ACOSolver(BaseSolver):
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
        os.remove(problem_filename)
        os.remove(parameter_filename)
        os.remove(tour_filename)
        
        aco = AntColonyTSP()
        aco_tour, len = aco.solve(scaled_distance_matrix, initial_tour=tour)

        return aco_tour

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == '__main__':
    n_nodes = 100  # Adjust as needed
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = ACOSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    if route is None:
        print(f"{solver.__class__.__name__} No solution found.")
    else:
        print(f"{solver.__class__.__name__} Best Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken: {time.time() - start_time}")