"""
Hypergraph processing module for Wolfram Physics Project.
Provides core functionality for creating, manipulating, and analyzing hypergraphs.
"""

import hypernetx as hnx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union
import copy


class HypergraphProcessor:
    """
    Core class for processing hypergraphs in the Wolfram Physics framework.
    Handles creation, manipulation, and analysis of evolving hypergraph structures.
    """
    
    def __init__(self, edges: Optional[Dict[str, List]] = None):
        """
        Initialize hypergraph processor.
        
        Args:
            edges: Dictionary mapping edge IDs to lists of nodes
        """
        self.hypergraph = hnx.Hypergraph(edges) if edges else hnx.Hypergraph()
        self.evolution_history = []
        self.current_step = 0
        
    @property
    def nodes(self) -> Set:
        """Get all nodes in the hypergraph."""
        return set(self.hypergraph.nodes)
    
    @property
    def edges(self) -> Dict:
        """Get all edges in the hypergraph."""
        return {edge: list(self.hypergraph.edges[edge]) for edge in self.hypergraph.edges}
    
    @property
    def node_count(self) -> int:
        """Get number of nodes."""
        return len(self.hypergraph.nodes)
    
    @property
    def edge_count(self) -> int:
        """Get number of edges."""
        return len(self.hypergraph.edges)
    
    def add_edge(self, edge_id: str, nodes: List[Union[str, int]]) -> None:
        """
        Add a new hyperedge to the graph.
        
        Args:
            edge_id: Unique identifier for the edge
            nodes: List of nodes that the edge connects
        """
        current_edges = self.edges
        current_edges[edge_id] = nodes
        self.hypergraph = hnx.Hypergraph(current_edges)
    
    def remove_edge(self, edge_id: str) -> None:
        """
        Remove a hyperedge from the graph.
        
        Args:
            edge_id: Identifier of the edge to remove
        """
        if edge_id in self.hypergraph.edges:
            current_edges = self.edges
            del current_edges[edge_id]
            self.hypergraph = hnx.Hypergraph(current_edges)
    
    def get_node_degree(self, node: Union[str, int]) -> int:
        """
        Get the degree of a node (number of edges it participates in).
        
        Args:
            node: Node identifier
            
        Returns:
            Degree of the node
        """
        return len([edge for edge in self.edges.values() if node in edge])
    
    def get_edge_size(self, edge_id: str) -> int:
        """
        Get the size of an edge (number of nodes it connects).
        
        Args:
            edge_id: Edge identifier
            
        Returns:
            Size of the edge
        """
        if edge_id in self.hypergraph.edges:
            return len(self.hypergraph.edges[edge_id])
        return 0
    
    def find_neighbors(self, node: Union[str, int]) -> Set:
        """
        Find all neighbors of a node (nodes that share an edge).
        
        Args:
            node: Node identifier
            
        Returns:
            Set of neighboring nodes
        """
        neighbors = set()
        for edge_nodes in self.edges.values():
            if node in edge_nodes:
                neighbors.update(edge_nodes)
        neighbors.discard(node)  # Remove the node itself
        return neighbors
    
    def get_connected_components(self) -> List[Set]:
        """
        Find all connected components in the hypergraph.
        
        Returns:
            List of sets, each containing nodes in a connected component
        """
        visited = set()
        components = []
        
        def dfs(node, component):
            if node in visited:
                return
            visited.add(node)
            component.add(node)
            for neighbor in self.find_neighbors(node):
                dfs(neighbor, component)
        
        for node in self.hypergraph.nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)
        
        return components
    
    def compute_clustering_coefficient(self) -> float:
        """
        Compute the clustering coefficient of the hypergraph.
        
        Returns:
            Clustering coefficient value
        """
        if self.node_count < 2:
            return 0.0
        
        total_clustering = 0.0
        
        for node in self.hypergraph.nodes:
            neighbors = self.find_neighbors(node)
            if len(neighbors) < 2:
                continue
            
            # Count edges between neighbors
            neighbor_edges = 0
            for edge_nodes in self.edges.values():
                edge_set = set(edge_nodes)
                if len(edge_set.intersection(neighbors)) >= 2:
                    neighbor_edges += 1
            
            # Possible edges between neighbors
            possible_edges = len(neighbors) * (len(neighbors) - 1) // 2
            
            if possible_edges > 0:
                total_clustering += neighbor_edges / possible_edges
        
        return total_clustering / self.node_count if self.node_count > 0 else 0.0
    
    def snapshot(self) -> Dict:
        """
        Take a snapshot of the current hypergraph state.
        
        Returns:
            Dictionary representation of the hypergraph
        """
        return {
            'edges': self.edges,
            'nodes': list(self.hypergraph.nodes),
            'step': self.current_step,
            'node_count': self.node_count,
            'edge_count': self.edge_count
        }
    
    def save_to_history(self) -> None:
        """Save current state to evolution history."""
        self.evolution_history.append(self.snapshot())
    
    def load_from_snapshot(self, snapshot: Dict) -> None:
        """
        Load hypergraph state from a snapshot.
        
        Args:
            snapshot: Dictionary containing hypergraph state
        """
        self.hypergraph = hnx.Hypergraph(snapshot['edges'])
        self.current_step = snapshot['step']
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix representation of the hypergraph.
        
        Returns:
            Adjacency matrix as numpy array
        """
        nodes = list(self.hypergraph.nodes)
        n = len(nodes)
        
        if n == 0:
            return np.array([])
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        adjacency = np.zeros((n, n), dtype=int)
        
        for edge_nodes in self.edges.values():
            edge_indices = [node_to_idx[node] for node in edge_nodes if node in node_to_idx]
            
            # Create connections between all pairs in the hyperedge
            for i in range(len(edge_indices)):
                for j in range(i + 1, len(edge_indices)):
                    adjacency[edge_indices[i]][edge_indices[j]] = 1
                    adjacency[edge_indices[j]][edge_indices[i]] = 1
        
        return adjacency
    
    def get_incidence_matrix(self) -> np.ndarray:
        """
        Get the incidence matrix representation of the hypergraph.
        
        Returns:
            Incidence matrix as numpy array
        """
        nodes = list(self.hypergraph.nodes)
        edges = list(self.edges.keys())
        
        if len(nodes) == 0 or len(edges) == 0:
            return np.array([])
        
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        incidence = np.zeros((len(nodes), len(edges)), dtype=int)
        
        for j, edge_id in enumerate(edges):
            edge_nodes = self.edges[edge_id]
            for node in edge_nodes:
                if node in node_to_idx:
                    incidence[node_to_idx[node]][j] = 1
        
        return incidence
    
    def compute_statistics(self) -> Dict:
        """
        Compute various statistics about the hypergraph.
        
        Returns:
            Dictionary containing statistical measures
        """
        stats = {
            'node_count': self.node_count,
            'edge_count': self.edge_count,
            'clustering_coefficient': self.compute_clustering_coefficient(),
            'connected_components': len(self.get_connected_components()),
            'evolution_steps': len(self.evolution_history)
        }
        
        if self.node_count > 0:
            degrees = [self.get_node_degree(node) for node in self.hypergraph.nodes]
            stats.update({
                'average_degree': np.mean(degrees),
                'max_degree': max(degrees),
                'min_degree': min(degrees),
                'degree_std': np.std(degrees)
            })
        
        if self.edge_count > 0:
            edge_sizes = [self.get_edge_size(edge_id) for edge_id in self.edges]
            stats.update({
                'average_edge_size': np.mean(edge_sizes),
                'max_edge_size': max(edge_sizes),
                'min_edge_size': min(edge_sizes),
                'edge_size_std': np.std(edge_sizes)
            })
        
        return stats
    
    def copy(self) -> 'HypergraphProcessor':
        """
        Create a deep copy of the hypergraph processor.
        
        Returns:
            New HypergraphProcessor instance with copied state
        """
        new_processor = HypergraphProcessor(self.edges)
        new_processor.evolution_history = copy.deepcopy(self.evolution_history)
        new_processor.current_step = self.current_step
        return new_processor
    
    def __str__(self) -> str:
        """String representation of the hypergraph."""
        return f"HypergraphProcessor(nodes={self.node_count}, edges={self.edge_count}, step={self.current_step})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()