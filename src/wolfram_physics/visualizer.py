"""
Visualization module for Wolfram Physics Project.
Provides interactive and static visualization capabilities for hypergraphs.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import hypernetx as hnx
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, Button, Slider, Select, Div, ColumnDataSource
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.colors import Color
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from .hypergraph_processor import HypergraphProcessor


class BasicVisualizer:
    """
    Core visualization class for hypergraphs in the Wolfram Physics framework.
    Supports both static (matplotlib) and interactive (bokeh) visualizations.
    """
    
    def __init__(self, processor: HypergraphProcessor):
        """
        Initialize the visualizer.
        
        Args:
            processor: HypergraphProcessor to visualize
        """
        self.processor = processor
        self.node_positions = {}
        self.color_scheme = {
            'nodes': '#1f77b4',
            'edges': '#ff7f0e',
            'hyperedges': '#2ca02c',
            'background': '#ffffff'
        }
        self.figure_size = (12, 8)
        
    def compute_layout(self, layout_type: str = 'spring') -> Dict:
        """
        Compute node positions using various layout algorithms.
        
        Args:
            layout_type: Type of layout ('spring', 'circular', 'random')
            
        Returns:
            Dictionary mapping nodes to (x, y) positions
        """
        if not self.processor.nodes:
            return {}
        
        # Convert hypergraph to regular graph for layout computation
        G = nx.Graph()
        G.add_nodes_from(self.processor.nodes)
        
        # Add edges based on hyperedge connections
        for edge_nodes in self.processor.edges.values():
            if len(edge_nodes) >= 2:
                # Create pairwise connections within hyperedges
                for i in range(len(edge_nodes)):
                    for j in range(i + 1, len(edge_nodes)):
                        G.add_edge(edge_nodes[i], edge_nodes[j])
        
        # Compute layout
        if layout_type == 'spring':
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout_type == 'circular':
            pos = nx.circular_layout(G)
        elif layout_type == 'random':
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        self.node_positions = pos
        return pos
    
    def plot_static(self, title: str = "Wolfram Physics Hypergraph", 
                   layout_type: str = 'spring', save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a static matplotlib visualization.
        
        Args:
            title: Title for the plot
            layout_type: Layout algorithm to use
            save_path: Path to save the figure (optional)
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        if not self.processor.nodes:
            ax.text(0.5, 0.5, 'Empty Hypergraph', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=16)
            ax.set_title(title)
            return fig
        
        # Compute layout
        pos = self.compute_layout(layout_type)
        
        # Draw hyperedges as polygons
        for edge_id, edge_nodes in self.processor.edges.items():
            if len(edge_nodes) >= 3:
                # Create polygon for hyperedges with 3+ nodes
                edge_positions = [pos[node] for node in edge_nodes if node in pos]
                if len(edge_positions) >= 3:
                    polygon_points = np.array(edge_positions)
                    polygon = patches.Polygon(polygon_points, alpha=0.3, 
                                            facecolor=self.color_scheme['hyperedges'],
                                            edgecolor='black', linewidth=1)
                    ax.add_patch(polygon)
            elif len(edge_nodes) == 2:
                # Draw regular edges
                node1, node2 = edge_nodes[0], edge_nodes[1]
                if node1 in pos and node2 in pos:
                    x_coords = [pos[node1][0], pos[node2][0]]
                    y_coords = [pos[node1][1], pos[node2][1]]
                    ax.plot(x_coords, y_coords, color=self.color_scheme['edges'], 
                           linewidth=2, alpha=0.7)
        
        # Draw nodes
        for node in self.processor.nodes:
            if node in pos:
                ax.scatter(pos[node][0], pos[node][1], 
                          c=self.color_scheme['nodes'], s=200, 
                          edgecolor='black', linewidth=2, zorder=5)
                ax.annotate(str(node), pos[node], xytext=(5, 5), 
                           textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add statistics
        stats = self.processor.compute_statistics()
        stats_text = f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}, Step: {self.processor.current_step}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_plot(self, title: str = "Interactive Wolfram Physics Hypergraph",
                               width: int = 800, height: int = 600) -> figure:
        """
        Create an interactive Bokeh visualization.
        
        Args:
            title: Title for the plot
            width: Plot width in pixels
            height: Plot height in pixels
            
        Returns:
            Bokeh figure object
        """
        p = figure(title=title, width=width, height=height,
                  tools="pan,wheel_zoom,box_zoom,reset,save",
                  toolbar_location="above")
        
        if not self.processor.nodes:
            p.text([0.5], [0.5], text=["Empty Hypergraph"], 
                  text_align="center", text_baseline="middle")
            return p
        
        # Compute layout
        pos = self.compute_layout('spring')
        
        # Prepare data for nodes
        node_x = [pos[node][0] for node in self.processor.nodes if node in pos]
        node_y = [pos[node][1] for node in self.processor.nodes if node in pos]
        node_labels = [str(node) for node in self.processor.nodes if node in pos]
        node_degrees = [self.processor.get_node_degree(node) for node in self.processor.nodes if node in pos]
        
        # Create node data source
        node_source = ColumnDataSource(data=dict(
            x=node_x,
            y=node_y,
            labels=node_labels,
            degrees=node_degrees,
            colors=[self.color_scheme['nodes']] * len(node_x)
        ))
        
        # Draw edges
        for edge_id, edge_nodes in self.processor.edges.items():
            if len(edge_nodes) == 2:
                # Regular edges
                node1, node2 = edge_nodes[0], edge_nodes[1]
                if node1 in pos and node2 in pos:
                    p.line([pos[node1][0], pos[node2][0]], 
                          [pos[node1][1], pos[node2][1]],
                          color=self.color_scheme['edges'], line_width=2, alpha=0.7)
            elif len(edge_nodes) >= 3:
                # Hyperedges as patches
                edge_positions = [pos[node] for node in edge_nodes if node in pos]
                if len(edge_positions) >= 3:
                    edge_x = [pos[0] for pos in edge_positions]
                    edge_y = [pos[1] for pos in edge_positions]
                    p.patch(edge_x, edge_y, alpha=0.3, 
                           color=self.color_scheme['hyperedges'], line_color="black")
        
        # Draw nodes
        node_glyphs = p.circle('x', 'y', size=15, color='colors', 
                              alpha=0.8, line_color="black", line_width=2, 
                              source=node_source)
        
        # Add node labels
        p.text('x', 'y', text='labels', text_align="center", 
              text_baseline="middle", text_font_size="10pt", 
              text_font_style="bold", source=node_source)
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Node", "@labels"),
            ("Degree", "@degrees"),
            ("Position", "(@x, @y)")
        ], renderers=[node_glyphs])
        p.add_tools(hover)
        
        # Style the plot
        p.title.text_font_size = "16pt"
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False
        
        return p
    
    def create_evolution_animation(self, evolution_history: List[Dict], 
                                 interval: int = 500, save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create an animated visualization of hypergraph evolution.
        
        Args:
            evolution_history: List of hypergraph snapshots
            interval: Animation interval in milliseconds
            save_path: Path to save animation (optional)
            
        Returns:
            matplotlib FuncAnimation object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        def animate(frame):
            ax.clear()
            
            if frame < len(evolution_history):
                snapshot = evolution_history[frame]
                
                # Create temporary processor for this frame
                temp_processor = HypergraphProcessor(snapshot['edges'])
                temp_processor.current_step = snapshot['step']
                
                # Update visualizer
                old_processor = self.processor
                self.processor = temp_processor
                
                # Compute layout (use consistent layout)
                pos = self.compute_layout('spring')
                
                # Draw hyperedges
                for edge_id, edge_nodes in snapshot['edges'].items():
                    if len(edge_nodes) >= 3:
                        edge_positions = [pos[node] for node in edge_nodes if node in pos]
                        if len(edge_positions) >= 3:
                            polygon_points = np.array(edge_positions)
                            polygon = patches.Polygon(polygon_points, alpha=0.3, 
                                                    facecolor=self.color_scheme['hyperedges'],
                                                    edgecolor='black', linewidth=1)
                            ax.add_patch(polygon)
                    elif len(edge_nodes) == 2:
                        node1, node2 = edge_nodes[0], edge_nodes[1]
                        if node1 in pos and node2 in pos:
                            x_coords = [pos[node1][0], pos[node2][0]]
                            y_coords = [pos[node1][1], pos[node2][1]]
                            ax.plot(x_coords, y_coords, color=self.color_scheme['edges'], 
                                   linewidth=2, alpha=0.7)
                
                # Draw nodes
                for node in snapshot['nodes']:
                    if node in pos:
                        ax.scatter(pos[node][0], pos[node][1], 
                                  c=self.color_scheme['nodes'], s=200, 
                                  edgecolor='black', linewidth=2, zorder=5)
                        ax.annotate(str(node), pos[node], xytext=(5, 5), 
                                   textcoords='offset points', fontsize=10, fontweight='bold')
                
                # Restore original processor
                self.processor = old_processor
                
                # Add frame info
                ax.set_title(f"Evolution Step {snapshot['step']}")
                ax.text(0.02, 0.98, f"Nodes: {snapshot['node_count']}, Edges: {snapshot['edge_count']}", 
                       transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       verticalalignment='top', fontsize=10)
            
            ax.set_aspect('equal')
            ax.axis('off')
        
        anim = FuncAnimation(fig, animate, frames=len(evolution_history), 
                           interval=interval, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=1000/interval)
        
        return anim
    
    def create_dashboard(self) -> Dict:
        """
        Create an interactive dashboard with controls.
        
        Returns:
            Dictionary containing dashboard components
        """
        # Create main plot
        plot = self.create_interactive_plot()
        
        # Create controls
        controls = {}
        
        # Layout selector
        controls['layout_select'] = Select(
            title="Layout Algorithm:",
            value="spring",
            options=[("spring", "Spring"), ("circular", "Circular"), ("random", "Random")]
        )
        
        # Statistics display
        stats = self.processor.compute_statistics()
        stats_text = f"""
        <h3>Hypergraph Statistics</h3>
        <p><strong>Nodes:</strong> {stats['node_count']}</p>
        <p><strong>Edges:</strong> {stats['edge_count']}</p>
        <p><strong>Evolution Steps:</strong> {stats['evolution_steps']}</p>
        <p><strong>Connected Components:</strong> {stats['connected_components']}</p>
        """
        if 'average_degree' in stats:
            stats_text += f"<p><strong>Average Degree:</strong> {stats['average_degree']:.2f}</p>"
        
        controls['stats_div'] = Div(text=stats_text)
        
        # Create layout
        dashboard = {
            'plot': plot,
            'controls': controls,
            'layout': row(column(controls['layout_select'], controls['stats_div']), plot)
        }
        
        return dashboard
    
    def export_static_image(self, filename: str, format: str = 'png', 
                           dpi: int = 300, layout_type: str = 'spring') -> None:
        """
        Export a static image of the hypergraph.
        
        Args:
            filename: Output filename
            format: Image format ('png', 'pdf', 'svg')
            dpi: Resolution for raster formats
            layout_type: Layout algorithm to use
        """
        fig = self.plot_static(layout_type=layout_type)
        fig.savefig(filename, format=format, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    def set_color_scheme(self, scheme: Dict[str, str]) -> None:
        """
        Set custom color scheme for visualization.
        
        Args:
            scheme: Dictionary with color definitions
        """
        self.color_scheme.update(scheme)
    
    def set_figure_size(self, width: int, height: int) -> None:
        """
        Set figure size for static plots.
        
        Args:
            width: Figure width in inches
            height: Figure height in inches
        """
        self.figure_size = (width, height)
    
    def __str__(self) -> str:
        """String representation of the visualizer."""
        return f"BasicVisualizer(nodes={self.processor.node_count}, edges={self.processor.edge_count})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()