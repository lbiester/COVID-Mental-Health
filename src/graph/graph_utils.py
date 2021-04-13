"""
Utility functions to support graph processing.
"""
from typing import Dict, Any

import numpy as np
import networkx as nx


def compute_graph_properties_general(graph: nx.Graph) -> Dict[str, Any]:
    """
    Compute properties of NetworkX graph (can be directed or undirected)
    :param graph: NetworkX graph
    :return: property_dict: dictionary mapping graph property names to values
    """
    property_dict = dict()
    property_dict["node_count"] = graph.number_of_nodes()
    property_dict["edge_count"] = graph.number_of_edges()
    property_dict["density"] = nx.density(graph)
    property_dict["mean_clustering_coefficient"] = nx.algorithms.average_clustering(graph)
    degrees = [graph.degree(node) for node in graph.nodes()]
    property_dict["mean_degree"] = np.mean(degrees)
    property_dict["degree_distr"] = np.histogram(degrees)
    if len(graph.edges):
        property_dict["assortativity"] = nx.degree_assortativity_coefficient(graph)
    else:
        property_dict["assortativity"] = np.NaN
    return property_dict


def compute_graph_properties_undirected(graph: nx.Graph) -> Dict[str, Any]:
    """
    Computer proerties of undirected graph.
    :param graph: NetworkX Undirected graph.
    :return: property_dict: dictionary mapping property names to values
    """
    property_dict = dict()
    property_dict["connected_component_count"] = nx.algorithms.number_connected_components(graph)
    _compute_distance_measures(graph, nx.connected_components, property_dict)
    general_property_dict = compute_graph_properties_general(graph)
    property_dict.update(general_property_dict)
    return property_dict


def compute_graph_properties_directed(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Compute properties of directed graph.
    :param graph: NetworkX Directed Graph
    :return: property_dict: dictionary mapping property names to values
    """
    property_dict = dict()
    reciprocal_graph = graph.to_undirected(reciprocal=True)
    property_dict["reciprocal_edge_count"] = reciprocal_graph.number_of_edges()
    property_dict["reciprocal_density"] = nx.density(reciprocal_graph)
    property_dict["connected_component_count"] = nx.algorithms.number_strongly_connected_components(graph)
    property_dict["mean_in_degree"] = np.mean([graph.in_degree(node) for node in graph.nodes()])
    property_dict["mean_out_degree"] = np.mean([graph.out_degree(node) for node in graph.nodes()])
    _compute_distance_measures(graph, nx.strongly_connected_components, property_dict)
    general_property_dict = compute_graph_properties_general(graph)
    property_dict.update(general_property_dict)
    return property_dict


def _compute_distance_measures(graph: nx.Graph, component_generator: Any, property_dict: Dict[str, Any]) -> None:
    component_sizes = [len(c) for c in component_generator(graph)]
    property_dict["mean_connected_component_size"] = np.mean(component_sizes)
    property_dict["connected_component_size_dist"] = np.histogram(component_sizes)
    # NOTE: when computing shortest paths, not considering whether is undirected or directed
    # so in directed case, both path a->b and b->a will be counted, meaning this path will have more weight than path
    # a -> a
    shortest_paths = [path_len for c in component_generator(graph)
                      for target_dict in dict(nx.shortest_path_length(graph.subgraph(c))).values()
                      for path_len in target_dict.values()]
    property_dict["mean_shortest_path_distr"] = np.mean(shortest_paths)
    property_dict["shortest_path_distr"] = np.histogram(shortest_paths)
    # get diameter: longest shortest path between any pair of nodes within a connect component
    property_dict["diameter"] = max(shortest_paths)

    # eccentricity: max distance of node v to any other node in network (e.g. how many hops away is furthest node?)
    eccentricities = [val for c in component_generator(graph)
                      for val in nx.algorithms.eccentricity(graph.subgraph(c)).values()]
    property_dict["mean_eccentricity"] = np.mean(eccentricities)
    property_dict["eccentricity_distr"] = np.histogram(eccentricities)
