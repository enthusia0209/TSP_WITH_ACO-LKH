import copy

from graphite.solvers import DPSolver, NearestNeighbourSolver, BeamSearchSolver, HPNSolver
from graphite.solvers.enhanced_greedy_solver import EnhancedNearestNeighbourSolver
from graphite.solvers.greedy_solver_vali import NearestNeighbourSolverVali
from graphite.solvers.or_solver import ORToolsSolver
from graphite.validator.reward import ScoreResponse
from graphite.solvers.lkh_solver import LKHSolver
from graphite.solvers.aco_solver import ACOSolver
from graphite.solvers.con_solver import CONSolver

solvers = {
    'small': DPSolver(),
    'large': NearestNeighbourSolver()
}
beam_solver = BeamSearchSolver()
nearest_neighbour_solver_vali = NearestNeighbourSolverVali()
hpn_solver = HPNSolver()
enhanced_solver = EnhancedNearestNeighbourSolver()
or_solver = ORToolsSolver()
lkh_solver = LKHSolver()
aco_solver = ACOSolver()
con_solver = CONSolver()

async def baseline_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    if new_synapse.problem.n_nodes < 15:
        # Solves the problem to optimality but is very computationally intensive
        route =  await solvers['small'].solve_problem(new_synapse.problem)
    else:
        # Simple heuristic that does not guarantee optimality.
        route =  await  solvers['large'].solve_problem(new_synapse.problem)
    new_synapse.solution = route
    # print(
    #     f"Miner returned value {synapse.solution}   length =  {len(synapse.solution) if isinstance(synapse.solution, list) else synapse.solution}"
    # )
    return new_synapse

async def beam_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  beam_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

async def nns_vali_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  nearest_neighbour_solver_vali.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

async def hpn_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  hpn_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

def scoring_solution(synapse_req):
    score_response_obj = ScoreResponse(synapse_req)
    miner_scores = score_response_obj.get_score(synapse_req)
    return miner_scores

async def enhanced_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  enhanced_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

async def or_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  or_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

async def lkh_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  lkh_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

async def aco_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  aco_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse

async def con_solver_solution(synapse):
    new_synapse = copy.deepcopy(synapse)
    route =  await  con_solver.solve_problem(new_synapse.problem)
    new_synapse.solution = route
    return new_synapse