# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


from typing import List, Union
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
import numpy as np  # type: ignore
import time
import asyncio

class EnhancedNearestNeighbourSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=250), GraphProblem(n_nodes=250, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem: List[List[Union[int, float]]], future_id: int, directed:bool) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])
        best_route = None
        shortest_distance = float('inf')

        # Try starting from each city
        for start_node in range(n):
            visited = [False] * n
            route = [start_node]
            visited[start_node] = True
            current_node = start_node
            total_distance = 0

            for _ in range(n - 1):
                nearest_distance = float('inf')
                nearest_node = -1

                for j in range(n):
                    if not visited[j] and distance_matrix[current_node][j] < nearest_distance:
                        nearest_distance = distance_matrix[current_node][j]
                        nearest_node = j

                if nearest_node == -1:  # If no unvisited node is found
                    break

                route.append(nearest_node)
                visited[nearest_node] = True
                total_distance += nearest_distance
                current_node = nearest_node

            # Return to the starting node
            total_distance += distance_matrix[current_node][route[0]]
            route.append(route[0])

            if total_distance < shortest_distance:
                shortest_distance = total_distance
                best_route = route

        return best_route

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == "__main__":
    # Runs the solver on a test MetricTSP
    n_nodes = 100  # Example number of nodes, adjust as needed
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = EnhancedNearestNeighbourSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    print(f"{solver.__class__.__name__} Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken for {n_nodes} Nodes: {time.time() - start_time}")