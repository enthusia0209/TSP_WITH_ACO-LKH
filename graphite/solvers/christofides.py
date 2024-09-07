import copy

import networkx as nx



def christofides_tsp(graph):
    # Step 1: Create a Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(graph)

    # Step 2: Find vertices with odd degrees in the MST
    odd_degree_nodes = [v for v, d in mst.degree() if d % 2 == 1]

    # Step 3: Find a minimum-weight perfect matching for the odd-degree vertices
    subgraph = graph.subgraph(odd_degree_nodes)
    matching = nx.algorithms.matching.max_weight_matching(subgraph, maxcardinality=True)

    # Step 4: Combine the MST and the matching to form a multigraph
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(matching)

    # Step 5: Find an Eulerian circuit in the multigraph
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))

    # Step 6: Shortcut the Eulerian circuit to form a Hamiltonian circuit (TSP tour)
    visited = set()
    tsp_tour = []
    for u, v in eulerian_circuit:
        if u not in visited:
            visited.add(u)
            tsp_tour.append(u)
    tsp_tour.append(tsp_tour[0])  # Return to starting node

    return tsp_tour

# Function to create a graph from a 2D weight matrix
def create_graph_from_matrix(num_nodes, weight_matrix):
    graph = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Ensure it's undirected and symmetric
            if weight_matrix[i][j] > 0:
                graph.add_edge(i, j, weight=weight_matrix[i][j])
    return graph

def solve(synapse_request):
    new_synapse = copy.deepcopy(synapse_request)
    num_nodes = synapse_request.problem.n_nodes
    weight_matrix = synapse_request.problem.edges
    # Create the graph using the weight matrix
    graph = create_graph_from_matrix(num_nodes, weight_matrix)

    # Run the Christofides algorithm
    tsp_tour = christofides_tsp(graph)
    new_synapse.solution = tsp_tour
    return new_synapse



