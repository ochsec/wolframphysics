"""
Test suite for HypergraphProcessor class.
"""

import pytest
import numpy as np
from src.wolfram_physics.hypergraph_processor import HypergraphProcessor


class TestHypergraphProcessor:
    """Test cases for HypergraphProcessor."""
    
    def test_initialization_empty(self):
        """Test initialization with empty hypergraph."""
        processor = HypergraphProcessor()
        assert processor.node_count == 0
        assert processor.edge_count == 0
        assert processor.current_step == 0
        assert len(processor.evolution_history) == 0
    
    def test_initialization_with_edges(self):
        """Test initialization with predefined edges."""
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C', 'D'],
            'e3': ['A', 'C']
        }
        processor = HypergraphProcessor(edges)
        assert processor.node_count == 4  # A, B, C, D
        assert processor.edge_count == 3
        assert set(processor.nodes) == {'A', 'B', 'C', 'D'}
    
    def test_add_edge(self):
        """Test adding edges to hypergraph."""
        processor = HypergraphProcessor()
        processor.add_edge('e1', ['A', 'B'])
        
        assert processor.node_count == 2
        assert processor.edge_count == 1
        assert 'A' in processor.nodes
        assert 'B' in processor.nodes
        assert 'e1' in processor.edges
    
    def test_remove_edge(self):
        """Test removing edges from hypergraph."""
        edges = {'e1': ['A', 'B'], 'e2': ['B', 'C']}
        processor = HypergraphProcessor(edges)
        
        processor.remove_edge('e1')
        assert processor.edge_count == 1
        assert 'e1' not in processor.edges
        assert 'e2' in processor.edges
    
    def test_node_degree(self):
        """Test node degree calculation."""
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C'],
            'e3': ['A', 'C', 'D']
        }
        processor = HypergraphProcessor(edges)
        
        assert processor.get_node_degree('A') == 2  # in e1 and e3
        assert processor.get_node_degree('B') == 2  # in e1 and e2
        assert processor.get_node_degree('C') == 2  # in e2 and e3
        assert processor.get_node_degree('D') == 1  # in e3 only
    
    def test_edge_size(self):
        """Test edge size calculation."""
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C', 'D'],
            'e3': ['A', 'C', 'D', 'E']
        }
        processor = HypergraphProcessor(edges)
        
        assert processor.get_edge_size('e1') == 2
        assert processor.get_edge_size('e2') == 3
        assert processor.get_edge_size('e3') == 4
        assert processor.get_edge_size('nonexistent') == 0
    
    def test_find_neighbors(self):
        """Test finding node neighbors."""
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C'],
            'e3': ['A', 'C', 'D']
        }
        processor = HypergraphProcessor(edges)
        
        neighbors_A = processor.find_neighbors('A')
        assert neighbors_A == {'B', 'C', 'D'}
        
        neighbors_B = processor.find_neighbors('B')
        assert neighbors_B == {'A', 'C'}
    
    def test_connected_components(self):
        """Test connected components detection."""
        # Create disconnected hypergraph
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C'],
            'e3': ['D', 'E']  # Disconnected component
        }
        processor = HypergraphProcessor(edges)
        
        components = processor.get_connected_components()
        assert len(components) == 2
        
        # Check that components contain correct nodes
        component_nodes = [comp for comp in components]
        assert {'A', 'B', 'C'} in component_nodes
        assert {'D', 'E'} in component_nodes
    
    def test_adjacency_matrix(self):
        """Test adjacency matrix generation."""
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C']
        }
        processor = HypergraphProcessor(edges)
        
        adj_matrix = processor.get_adjacency_matrix()
        assert adj_matrix.shape == (3, 3)  # 3 nodes: A, B, C
        
        # Check that matrix is symmetric
        assert np.array_equal(adj_matrix, adj_matrix.T)
    
    def test_incidence_matrix(self):
        """Test incidence matrix generation."""
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C', 'D']
        }
        processor = HypergraphProcessor(edges)
        
        inc_matrix = processor.get_incidence_matrix()
        assert inc_matrix.shape == (4, 2)  # 4 nodes, 2 edges
        
        # Check that each edge has correct number of 1s
        assert np.sum(inc_matrix[:, 0]) == 2  # e1 has 2 nodes
        assert np.sum(inc_matrix[:, 1]) == 3  # e2 has 3 nodes
    
    def test_snapshot(self):
        """Test taking snapshots."""
        edges = {'e1': ['A', 'B'], 'e2': ['B', 'C']}
        processor = HypergraphProcessor(edges)
        
        snapshot = processor.snapshot()
        
        assert snapshot['node_count'] == 3
        assert snapshot['edge_count'] == 2
        assert snapshot['step'] == 0
        assert set(snapshot['nodes']) == {'A', 'B', 'C'}
        assert snapshot['edges'] == edges
    
    def test_save_and_load_history(self):
        """Test saving and loading from history."""
        processor = HypergraphProcessor()
        processor.add_edge('e1', ['A', 'B'])
        
        # Save to history
        processor.save_to_history()
        assert len(processor.evolution_history) == 1
        
        # Modify processor
        processor.add_edge('e2', ['B', 'C'])
        processor.current_step = 1
        
        # Load from history
        processor.load_from_snapshot(processor.evolution_history[0])
        assert processor.node_count == 2
        assert processor.edge_count == 1
        assert processor.current_step == 0
    
    def test_statistics(self):
        """Test statistics computation."""
        edges = {
            'e1': ['A', 'B'],
            'e2': ['B', 'C'],
            'e3': ['C', 'D']
        }
        processor = HypergraphProcessor(edges)
        
        stats = processor.compute_statistics()
        
        assert stats['node_count'] == 4
        assert stats['edge_count'] == 3
        assert stats['connected_components'] == 1
        assert 'average_degree' in stats
        assert 'average_edge_size' in stats
    
    def test_copy(self):
        """Test copying processor."""
        edges = {'e1': ['A', 'B'], 'e2': ['B', 'C']}
        processor = HypergraphProcessor(edges)
        processor.save_to_history()
        
        copy_processor = processor.copy()
        
        assert copy_processor.node_count == processor.node_count
        assert copy_processor.edge_count == processor.edge_count
        assert copy_processor.current_step == processor.current_step
        assert len(copy_processor.evolution_history) == len(processor.evolution_history)
        
        # Verify it's a deep copy
        copy_processor.add_edge('e3', ['D', 'E'])
        assert copy_processor.edge_count != processor.edge_count
    
    def test_string_representation(self):
        """Test string representation."""
        edges = {'e1': ['A', 'B']}
        processor = HypergraphProcessor(edges)
        
        str_repr = str(processor)
        assert 'HypergraphProcessor' in str_repr
        assert 'nodes=2' in str_repr
        assert 'edges=1' in str_repr
        assert 'step=0' in str_repr


class TestHypergraphProcessorEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_hypergraph_operations(self):
        """Test operations on empty hypergraph."""
        processor = HypergraphProcessor()
        
        assert processor.get_node_degree('A') == 0
        assert processor.find_neighbors('A') == set()
        assert processor.get_connected_components() == []
        assert processor.get_adjacency_matrix().size == 0
        assert processor.get_incidence_matrix().size == 0
    
    def test_single_node_operations(self):
        """Test operations with single node."""
        processor = HypergraphProcessor()
        processor.add_edge('e1', ['A'])
        
        assert processor.node_count == 1
        assert processor.get_node_degree('A') == 1
        assert processor.find_neighbors('A') == set()
        assert len(processor.get_connected_components()) == 1
    
    def test_large_hyperedge(self):
        """Test with large hyperedge."""
        nodes = [f'node_{i}' for i in range(100)]
        processor = HypergraphProcessor({'large_edge': nodes})
        
        assert processor.node_count == 100
        assert processor.edge_count == 1
        assert processor.get_edge_size('large_edge') == 100
    
    def test_duplicate_edges(self):
        """Test handling of duplicate edge additions."""
        processor = HypergraphProcessor()
        processor.add_edge('e1', ['A', 'B'])
        processor.add_edge('e1', ['C', 'D'])  # Overwrites previous
        
        assert processor.edge_count == 1
        assert processor.edges['e1'] == ['C', 'D']
    
    def test_removing_nonexistent_edge(self):
        """Test removing non-existent edge."""
        processor = HypergraphProcessor()
        processor.remove_edge('nonexistent')  # Should not raise error
        assert processor.edge_count == 0


if __name__ == '__main__':
    pytest.main([__file__])