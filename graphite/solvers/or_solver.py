from typing import List
from graphite.solvers.base_solver import BaseSolver
from graphite.protocol import GraphProblem
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import asyncio
import time

class ORToolsSolver(BaseSolver):
    def __init__(self, problem_types: List[GraphProblem] = [GraphProblem(n_nodes=2), GraphProblem(n_nodes=2, directed=True, problem_type='General TSP')]):
        super().__init__(problem_types=problem_types)

    async def solve(self, formatted_problem, future_id: int, directed:bool) -> List[int]:
        distance_matrix = formatted_problem
        n = len(distance_matrix[0])

        # Determine if the matrix contains float values
        is_float = isinstance(distance_matrix[0][0], float)

        # Scale factor for converting float distances to integers if necessary
        scale_factor = 1000 if is_float else 1

        # Scale the distance matrix if necessary
        scaled_distance_matrix = [
            [int(round(distance * scale_factor)) for distance in row]
            for row in distance_matrix
        ]

        # Create the routing index manager
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)

        # Create Routing Model
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            """Returns the distance between the two nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return scaled_distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC
        search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_parameters.time_limit.seconds = 3  # Allow a reasonable time for finding a solution

        # Solve the problem
        solution = routing.SolveWithParameters(search_parameters)

        if not solution:
            return None  # Indicate that no solution was found

        # Get the tour from the solution
        index = routing.Start(0)
        tour = []
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        tour.append(tour[0])  # to make it a complete tour

        # Evaluate the cost of the current solution
        current_cost = sum(scaled_distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

        # Convert the cost back to float if necessary
        if is_float:
            current_cost = current_cost / scale_factor

        return tour

    def problem_transformations(self, problem: GraphProblem):
        return problem.edges

if __name__ == '__main__':
    n_nodes = 100  # Adjust as needed
    test_problem = GraphProblem(n_nodes=n_nodes)
    solver = ORToolsSolver(problem_types=[test_problem.problem_type])
    start_time = time.time()
    route = asyncio.run(solver.solve_problem(test_problem))
    if route is None:
        print(f"{solver.__class__.__name__} No solution found.")
    else:
        print(f"{solver.__class__.__name__} Best Solution: {route}")
    print(f"{solver.__class__.__name__} Time Taken: {time.time() - start_time}")