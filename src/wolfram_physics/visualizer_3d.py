"""
3D visualization module for Wolfram Physics Project.
Provides 3D interactive visualization capabilities using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

from .hypergraph_processor import HypergraphProcessor
from .visualizer import BasicVisualizer


class Visualizer3D:
    """
    3D visualization class for hypergraphs using Plotly.
    Provides interactive 3D views with rotation, zoom, and selection.
    """
    
    def __init__(self, processor: HypergraphProcessor):
        """
        Initialize 3D visualizer.
        
        Args:
            processor: HypergraphProcessor to visualize
        """
        self.processor = processor
        self.node_positions_3d = {}
        self.layout_algorithm = 'spring_3d'
        self.color_map = px.colors.qualitative.Plotly
        
    def compute_3d_layout(self, algorithm: str = 'spring_3d') -> Dict:
        """
        Compute 3D node positions using various algorithms.
        
        Args:
            algorithm: Layout algorithm ('spring_3d', 'mds', 'random_3d', 'spherical')
            
        Returns:
            Dictionary mapping nodes to (x, y, z) positions
        """
        if not self.processor.nodes:
            return {}
        
        nodes = list(self.processor.nodes)
        n = len(nodes)
        
        if algorithm == 'spring_3d':
            # Create NetworkX graph for layout
            G = nx.Graph()
            G.add_nodes_from(nodes)
            
            # Add edges based on hyperedge connections
            for edge_nodes in self.processor.edges.values():
                if len(edge_nodes) >= 2:
                    for i in range(len(edge_nodes)):
                        for j in range(i + 1, len(edge_nodes)):
                            G.add_edge(edge_nodes[i], edge_nodes[j])
            
            # Use spectral layout as initial positions for better results
            try:
                pos_2d = nx.spectral_layout(G, dim=2)
                # Extend to 3D with random z-coordinates
                pos_3d = {}
                for node, (x, y) in pos_2d.items():
                    pos_3d[node] = (x, y, np.random.uniform(-0.5, 0.5))
            except:
                # Fallback to random if spectral fails
                pos_3d = {node: (np.random.uniform(-1, 1), 
                                np.random.uniform(-1, 1),
                                np.random.uniform(-1, 1)) for node in nodes}
            
            # Apply spring layout iterations
            pos_3d = self._spring_layout_3d(G, pos_3d, iterations=50)
            
        elif algorithm == 'mds':
            # Multi-dimensional scaling based on graph distances
            pos_3d = self._mds_layout_3d()
            
        elif algorithm == 'spherical':
            # Arrange nodes on a sphere
            pos_3d = self._spherical_layout()
            
        else:  # random_3d
            pos_3d = {node: (np.random.uniform(-1, 1), 
                           np.random.uniform(-1, 1),
                           np.random.uniform(-1, 1)) for node in nodes}
        
        self.node_positions_3d = pos_3d
        return pos_3d
    
    def _spring_layout_3d(self, G: nx.Graph, pos: Dict, 
                         iterations: int = 50, k: float = None) -> Dict:
        """
        3D spring layout algorithm.
        
        Args:
            G: NetworkX graph
            pos: Initial positions
            iterations: Number of iterations
            k: Spring constant
            
        Returns:
            Updated positions
        """
        if k is None:
            k = 1.0 / np.sqrt(len(G.nodes()))
        
        nodes = list(G.nodes())
        
        for _ in range(iterations):
            # Calculate forces
            forces = {node: np.array([0.0, 0.0, 0.0]) for node in nodes}
            
            # Repulsive forces between all nodes
            for i, u in enumerate(nodes):
                for j, v in enumerate(nodes[i+1:], i+1):
                    delta = np.array(pos[u]) - np.array(pos[v])
                    distance = np.linalg.norm(delta)
                    if distance > 0.001:  # Avoid division by very small numbers
                        force = min(k * k / distance, 10.0)  # Cap force to prevent overflow
                        forces[u] += force * delta / distance
                        forces[v] -= force * delta / distance
            
            # Attractive forces for edges
            for u, v in G.edges():
                delta = np.array(pos[v]) - np.array(pos[u])
                distance = np.linalg.norm(delta)
                if distance > 0.001:  # Avoid division by very small numbers
                    force = min(distance * distance / k, 10.0)  # Cap force
                    forces[u] += force * delta / distance
                    forces[v] -= force * delta / distance
            
            # Update positions
            for node in nodes:
                pos[node] = tuple(np.array(pos[node]) + forces[node] * 0.1)
        
        return pos
    
    def _mds_layout_3d(self) -> Dict:
        """
        Multi-dimensional scaling layout in 3D.
        
        Returns:
            Node positions dictionary
        """
        nodes = list(self.processor.nodes)
        n = len(nodes)
        
        if n < 3:
            return {node: (i, 0, 0) for i, node in enumerate(nodes)}
        
        # Compute shortest path distances
        G = nx.Graph()
        G.add_nodes_from(nodes)
        
        for edge_nodes in self.processor.edges.values():
            if len(edge_nodes) >= 2:
                for i in range(len(edge_nodes)):
                    for j in range(i + 1, len(edge_nodes)):
                        G.add_edge(edge_nodes[i], edge_nodes[j])
        
        # Get distance matrix
        distances = np.zeros((n, n))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if i != j:
                    try:
                        distances[i, j] = nx.shortest_path_length(G, u, v)
                    except nx.NetworkXNoPath:
                        distances[i, j] = n  # Max distance for disconnected
        
        # Apply MDS
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        positions = mds.fit_transform(distances)
        
        # Normalize positions
        scaler = StandardScaler()
        positions = scaler.fit_transform(positions)
        
        return {node: tuple(positions[i]) for i, node in enumerate(nodes)}
    
    def _spherical_layout(self) -> Dict:
        """
        Arrange nodes on a sphere surface.
        
        Returns:
            Node positions dictionary
        """
        nodes = list(self.processor.nodes)
        n = len(nodes)
        
        positions = {}
        
        # Use golden angle for even distribution
        golden_angle = np.pi * (3 - np.sqrt(5))
        
        for i, node in enumerate(nodes):
            theta = golden_angle * i
            y = 1 - (i / float(n - 1)) * 2  # -1 to 1
            radius = np.sqrt(1 - y * y)
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            positions[node] = (x, y, z)
        
        return positions
    
    def create_3d_plot(self, title: str = "3D Wolfram Physics Hypergraph",
                      layout: str = 'spring_3d',
                      show_edges: bool = True,
                      show_hyperedges: bool = True,
                      node_size: int = 8,
                      width: int = 800,
                      height: int = 600) -> go.Figure:
        """
        Create an interactive 3D visualization.
        
        Args:
            title: Plot title
            layout: Layout algorithm
            show_edges: Whether to show regular edges
            show_hyperedges: Whether to show hyperedges
            node_size: Size of nodes
            width: Figure width
            height: Figure height
            
        Returns:
            Plotly Figure object
        """
        # Compute layout
        pos_3d = self.compute_3d_layout(layout)
        
        if not pos_3d:
            # Empty hypergraph
            fig = go.Figure()
            fig.add_annotation(
                text="Empty Hypergraph",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Create figure
        fig = go.Figure()
        
        # Draw edges
        if show_edges:
            edge_trace = self._create_edge_trace(pos_3d)
            if edge_trace:
                fig.add_trace(edge_trace)
        
        # Draw hyperedges
        if show_hyperedges:
            hyperedge_traces = self._create_hyperedge_traces(pos_3d)
            for trace in hyperedge_traces:
                fig.add_trace(trace)
        
        # Draw nodes
        node_trace = self._create_node_trace(pos_3d, node_size)
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            showlegend=False,
            scene=dict(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False),
                bgcolor='rgb(240, 240, 240)'
            ),
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode='closest'
        )
        
        return fig
    
    def _create_edge_trace(self, pos_3d: Dict) -> Optional[go.Scatter3d]:
        """Create 3D edge trace."""
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge_id, edge_nodes in self.processor.edges.items():
            if len(edge_nodes) == 2:
                node1, node2 = edge_nodes[0], edge_nodes[1]
                if node1 in pos_3d and node2 in pos_3d:
                    x0, y0, z0 = pos_3d[node1]
                    x1, y1, z1 = pos_3d[node2]
                    
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_z.extend([z0, z1, None])
        
        if not edge_x:
            return None
        
        return go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125,125,125,0.5)', width=2),
            hoverinfo='none'
        )
    
    def _create_hyperedge_traces(self, pos_3d: Dict) -> List[go.Scatter3d]:
        """Create 3D hyperedge traces as surfaces."""
        traces = []
        
        for edge_id, edge_nodes in self.processor.edges.items():
            if len(edge_nodes) >= 3:
                # Create surface mesh for hyperedge
                positions = [pos_3d[node] for node in edge_nodes if node in pos_3d]
                
                if len(positions) >= 3:
                    # Create triangulated surface
                    x_coords = [p[0] for p in positions]
                    y_coords = [p[1] for p in positions]
                    z_coords = [p[2] for p in positions]
                    
                    # Add center point for better visualization
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    center_z = np.mean(z_coords)
                    
                    # Create mesh3d trace
                    trace = go.Mesh3d(
                        x=x_coords + [center_x],
                        y=y_coords + [center_y],
                        z=z_coords + [center_z],
                        opacity=0.3,
                        color='lightblue',
                        alphahull=0,
                        hoverinfo='text',
                        text=f'Hyperedge {edge_id}'
                    )
                    traces.append(trace)
        
        return traces
    
    def _create_node_trace(self, pos_3d: Dict, node_size: int) -> go.Scatter3d:
        """Create 3D node trace."""
        node_x = []
        node_y = []
        node_z = []
        node_text = []
        node_colors = []
        
        for node in self.processor.nodes:
            if node in pos_3d:
                x, y, z = pos_3d[node]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                
                # Node info
                degree = self.processor.get_node_degree(node)
                neighbors = list(self.processor.find_neighbors(node))
                node_text.append(
                    f"Node: {node}<br>"
                    f"Degree: {degree}<br>"
                    f"Neighbors: {', '.join(map(str, neighbors[:5]))}"
                    f"{'...' if len(neighbors) > 5 else ''}"
                )
                
                # Color by degree
                node_colors.append(degree)
        
        return go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title="Degree",
                    tickmode='linear',
                    tick0=0,
                    dtick=1
                ),
                line=dict(color='black', width=1)
            ),
            text=[str(node) for node in self.processor.nodes if node in pos_3d],
            textposition="top center",
            textfont=dict(size=10, color='black'),
            hoverinfo='text',
            hovertext=node_text
        )
    
    def create_evolution_animation_3d(self, evolution_history: List[Dict],
                                    layout: str = 'spring_3d',
                                    duration: int = 500) -> go.Figure:
        """
        Create animated 3D visualization of hypergraph evolution.
        
        Args:
            evolution_history: List of hypergraph snapshots
            layout: Layout algorithm
            duration: Frame duration in milliseconds
            
        Returns:
            Plotly Figure with animation
        """
        if not evolution_history:
            return self.create_3d_plot()
        
        # Create frames
        frames = []
        
        for i, snapshot in enumerate(evolution_history):
            # Create temporary processor for this frame
            temp_processor = HypergraphProcessor(snapshot['edges'])
            temp_viz = Visualizer3D(temp_processor)
            
            # Compute layout
            pos_3d = temp_viz.compute_3d_layout(layout)
            
            # Create traces for this frame
            traces = []
            
            # Edges
            edge_trace = temp_viz._create_edge_trace(pos_3d)
            if edge_trace:
                traces.append(edge_trace)
            
            # Nodes
            node_trace = temp_viz._create_node_trace(pos_3d, 8)
            traces.append(node_trace)
            
            frame = go.Frame(
                data=traces,
                name=str(i),
                layout=go.Layout(title=f"Evolution Step {i}")
            )
            frames.append(frame)
        
        # Create initial figure
        temp_processor = HypergraphProcessor(evolution_history[0]['edges'])
        temp_viz = Visualizer3D(temp_processor)
        fig = temp_viz.create_3d_plot(title="Hypergraph Evolution (3D)")
        
        # Add frames
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': duration, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': duration/2}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[f.name], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }],
                        'label': f'Step {i}',
                        'method': 'animate'
                    }
                    for i, f in enumerate(fig.frames)
                ],
                'active': 0,
                'y': 0,
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        return fig
    
    def create_multi_view_3d(self, layouts: List[str] = None,
                           titles: List[str] = None) -> go.Figure:
        """
        Create multiple 3D views in subplots.
        
        Args:
            layouts: List of layout algorithms
            titles: List of subplot titles
            
        Returns:
            Plotly Figure with subplots
        """
        if layouts is None:
            layouts = ['spring_3d', 'mds', 'spherical']
        
        if titles is None:
            titles = ['Spring Layout', 'MDS Layout', 'Spherical Layout']
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=len(layouts),
            subplot_titles=titles,
            specs=[[{'type': 'scatter3d'}] * len(layouts)],
            horizontal_spacing=0.05
        )
        
        # Add each layout
        for i, (layout, title) in enumerate(zip(layouts, titles)):
            # Compute layout
            pos_3d = self.compute_3d_layout(layout)
            
            # Add traces
            edge_trace = self._create_edge_trace(pos_3d)
            if edge_trace:
                fig.add_trace(edge_trace, row=1, col=i+1)
            
            node_trace = self._create_node_trace(pos_3d, 6)
            fig.add_trace(node_trace, row=1, col=i+1)
        
        # Update layout
        fig.update_layout(
            title="Multiple 3D Views Comparison",
            showlegend=False,
            height=400,
            width=1200
        )
        
        # Update scene for each subplot
        for i in range(len(layouts)):
            fig.update_scenes(
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, zeroline=False, visible=False),
                bgcolor='rgb(245, 245, 245)',
                row=1, col=i+1
            )
        
        return fig
    
    def export_3d_html(self, filename: str, **kwargs) -> None:
        """
        Export 3D visualization as interactive HTML.
        
        Args:
            filename: Output filename
            **kwargs: Additional arguments for create_3d_plot
        """
        fig = self.create_3d_plot(**kwargs)
        fig.write_html(filename, include_plotlyjs='cdn')
    
    def __str__(self) -> str:
        """String representation."""
        return f"Visualizer3D(nodes={self.processor.node_count}, edges={self.processor.edge_count})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()