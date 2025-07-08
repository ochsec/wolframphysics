"""
Interactive visualization module for Wolfram Physics Project.
Provides real-time, interactive visualization with Bokeh server and enhanced UI controls.
"""

import numpy as np
from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, HoverTool, Button, Slider, Select, TextInput,
    Div, Panel, Tabs, CheckboxGroup, RadioGroup, DataTable, TableColumn,
    CustomJS, Toggle, RangeSlider, ColorPicker
)
from bokeh.layouts import column, row, layout, gridplot
from bokeh.events import Tap, DoubleTap
from bokeh.palettes import Spectral, Viridis, Category20
from bokeh.transform import linear_cmap
from bokeh.models.tools import BoxSelectTool, LassoSelectTool, ResetTool
import asyncio
from functools import partial
from typing import Dict, List, Optional, Tuple, Callable, Any
import json
import time
from datetime import datetime

from .hypergraph_processor import HypergraphProcessor
from .rule_engine import WolframRuleEngine, RewriteRule
from .visualizer import BasicVisualizer
from .data_manager import DataManager


class InteractiveVisualizer:
    """
    Advanced interactive visualization for Wolfram Physics hypergraphs.
    Provides real-time controls, multi-view displays, and server-based interactivity.
    """
    
    def __init__(self, initial_processor: Optional[HypergraphProcessor] = None,
                 rule_engine: Optional[WolframRuleEngine] = None):
        """
        Initialize the interactive visualizer.
        
        Args:
            initial_processor: Initial HypergraphProcessor to visualize
            rule_engine: WolframRuleEngine for evolution control
        """
        self.processor = initial_processor or HypergraphProcessor()
        self.rule_engine = rule_engine or WolframRuleEngine()
        
        # Visualization state
        self.current_layout = 'spring'
        self.color_scheme = 'mathematica'
        self.node_size_factor = 25
        self.edge_width_factor = 4
        self.animation_speed = 500  # milliseconds
        
        # Enhanced styling for Mathematica-like appearance
        self.node_outline_width = 3
        self.edge_alpha = 0.9
        self.arrow_alpha = 0.95
        self.node_glow_effect = True
        self.edge_gradient = True
        self.background_color = "#FAFAFA"
        self.grid_alpha = 0.1
        
        # Evolution state
        self.is_evolving = False
        self.evolution_task = None
        self.evolution_history = []
        self.current_history_index = 0
        
        # Interactive state
        self.selected_nodes = set()
        self.selected_edges = set()
        self.highlight_mode = 'none'
        
        # UI components (initialized in create methods)
        self.plot = None
        self.node_source = None
        self.edge_sources = []
        self.controls = {}
        self.info_displays = {}
        
        # Callbacks
        self.evolution_callbacks = []
        self.selection_callbacks = []
        
    def create_main_plot(self, width: int = 800, height: int = 600) -> figure:
        """
        Create the main interactive plot.
        
        Args:
            width: Plot width in pixels
            height: Plot height in pixels
            
        Returns:
            Bokeh figure object
        """
        self.plot = figure(
            title="Interactive Wolfram Physics Hypergraph",
            width=width,
            height=height,
            tools="pan,wheel_zoom,box_zoom,reset,save,tap,box_select,lasso_select",
            toolbar_location="above",
            active_scroll="wheel_zoom",
            background_fill_color=self.background_color,
            border_fill_color="white"
        )
        
        # Initialize data sources
        self._update_data_sources()
        
        # Draw edges BEFORE nodes so they appear behind
        # Use the stored node positions for consistency
        if self.processor.edges and hasattr(self, 'node_positions'):
            pos = self.node_positions
            
            for edge_id, edge_nodes in self.processor.edges.items():
                if len(edge_nodes) == 2 and edge_nodes[0] in pos and edge_nodes[1] in pos:
                    x1, y1 = pos[edge_nodes[0]]
                    x2, y2 = pos[edge_nodes[1]]
                    
                    # Draw line
                    self.plot.line(
                        [x1, x2], [y1, y2],
                        line_width=self.edge_width_factor,
                        line_color="gray",
                        alpha=0.6
                    )
                    
                    # Add directional arrow if enabled
                    show_arrows = True
                    if 'show_options' in self.controls and hasattr(self.controls['show_options'], 'active'):
                        show_arrows = 3 in self.controls['show_options'].active
                    
                    if show_arrows:
                        self._add_arrow(x1, y1, x2, y2, edge_id)
        
        # Draw nodes
        node_renderer = self.plot.scatter(
            'x', 'y', size='size', color='color', alpha='alpha',
            line_color="black", line_width=2, source=self.node_source,
            selection_color="red", selection_alpha=1.0,
            nonselection_alpha=0.6
        )
        
        # Add node labels
        self.plot.text(
            'x', 'y', text='label', text_align="center",
            text_baseline="middle", text_font_size="10pt",
            source=self.node_source
        )
        
        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Node", "@label"),
                ("Degree", "@degree"),
                ("Neighbors", "@neighbors"),
                ("Position", "(@x, @y)")
            ],
            renderers=[node_renderer]
        )
        self.plot.add_tools(hover)
        
        # Enhanced plot styling
        self.plot.title.text_font_size = "18pt"
        self.plot.xaxis.visible = False
        self.plot.yaxis.visible = False
        self.plot.xgrid.visible = False
        self.plot.ygrid.visible = False
        self.plot.background_fill_color = self.background_color
        self.plot.outline_line_color = "#BDC3C7"
        self.plot.outline_line_width = 2
        
        # Add tap callback for node selection
        self.plot.on_event(Tap, self._on_tap)
        
        return self.plot
    
    def create_evolution_controls(self) -> layout:
        """
        Create evolution control panel.
        
        Returns:
            Bokeh layout with evolution controls
        """
        # Play/Pause button
        self.controls['play_pause'] = Toggle(
            label="▶ Play", 
            button_type="success",
            width=100
        )
        self.controls['play_pause'].on_click(self._toggle_evolution)
        
        # Step buttons
        self.controls['step_forward'] = Button(
            label="Step →", 
            button_type="primary",
            width=80
        )
        self.controls['step_forward'].on_click(self._step_forward)
        
        self.controls['step_backward'] = Button(
            label="← Step", 
            button_type="primary",
            width=80
        )
        self.controls['step_backward'].on_click(self._step_backward)
        
        # Reset button
        self.controls['reset'] = Button(
            label="Reset", 
            button_type="warning",
            width=80
        )
        self.controls['reset'].on_click(self._reset_evolution)
        
        # Speed slider
        self.controls['speed'] = Slider(
            title="Evolution Speed (ms)",
            start=100,
            end=2000,
            value=self.animation_speed,
            step=100,
            width=300
        )
        self.controls['speed'].on_change('value', self._update_speed)
        
        # Max applications per step
        self.controls['max_applications'] = Slider(
            title="Max Applications/Step",
            start=1,
            end=10,
            value=1,
            step=1,
            width=300
        )
        
        # Evolution info
        self.info_displays['evolution_info'] = Div(
            text=self._get_evolution_info(),
            width=300,
            height=80
        )
        
        # Layout
        controls_row = row(
            self.controls['play_pause'],
            self.controls['step_backward'],
            self.controls['step_forward'],
            self.controls['reset']
        )
        
        return column(
            Div(text="<h3>Evolution Controls</h3>"),
            controls_row,
            self.controls['speed'],
            self.controls['max_applications'],
            self.info_displays['evolution_info']
        )
    
    def create_rule_controls(self) -> layout:
        """
        Create rule editing and management controls.
        
        Returns:
            Bokeh layout with rule controls
        """
        # Rule selector
        rule_options = [(rule.name, rule.name) for rule in self.rule_engine.rules]
        self.controls['rule_select'] = Select(
            title="Active Rules:",
            value=rule_options[0][0] if rule_options else "",
            options=rule_options,
            width=300
        )
        
        # Rule priority
        self.controls['rule_priority'] = Slider(
            title="Rule Priority",
            start=1,
            end=10,
            value=1,
            step=1,
            width=300
        )
        
        # Add/Remove rule buttons
        self.controls['add_rule'] = Button(
            label="Add Rule",
            button_type="success",
            width=100
        )
        self.controls['add_rule'].on_click(self._show_add_rule_dialog)
        
        self.controls['remove_rule'] = Button(
            label="Remove Rule",
            button_type="danger",
            width=100
        )
        self.controls['remove_rule'].on_click(self._remove_selected_rule)
        
        # Predefined rule sets
        self.controls['rule_presets'] = Select(
            title="Load Rule Preset:",
            value="custom",
            options=[
                ("custom", "Custom"),
                ("basic", "Basic Rules"),
                ("advanced", "Advanced Rules"),
                ("physics", "Physics Rules")
            ],
            width=300
        )
        self.controls['rule_presets'].on_change('value', self._load_rule_preset)
        
        # Rule info display
        self.info_displays['rule_info'] = Div(
            text=self._get_rule_info(),
            width=300,
            height=150
        )
        
        # Layout
        return column(
            Div(text="<h3>Rule Management</h3>"),
            self.controls['rule_presets'],
            self.controls['rule_select'],
            self.controls['rule_priority'],
            row(self.controls['add_rule'], self.controls['remove_rule']),
            self.info_displays['rule_info']
        )
    
    def create_visualization_controls(self) -> layout:
        """
        Create visualization parameter controls.
        
        Returns:
            Bokeh layout with visualization controls
        """
        # Layout algorithm
        self.controls['layout'] = Select(
            title="Layout Algorithm:",
            value=self.current_layout,
            options=[
                ("spring", "Spring Layout"),
                ("circular", "Circular Layout"),
                ("hierarchical", "Hierarchical Layout"),
                ("random", "Random Layout"),
                ("spectral", "Spectral Layout")
            ],
            width=300
        )
        self.controls['layout'].on_change('value', self._update_layout)
        
        # Color scheme
        self.controls['color_scheme'] = Select(
            title="Color Scheme:",
            value=self.color_scheme,
            options=[
                ("mathematica", "Mathematica Style"),
                ("degree", "By Degree"),
                ("component", "By Component"),
                ("evolution", "By Evolution Time"),
                ("viridis", "Viridis"),
                ("plasma", "Plasma"),
                ("sunset", "Sunset Colors"),
                ("ocean", "Ocean Blues"),
                ("custom", "Custom")
            ],
            width=300
        )
        self.controls['color_scheme'].on_change('value', self._update_colors)
        
        # Node size
        self.controls['node_size'] = Slider(
            title="Node Size",
            start=5,
            end=50,
            value=self.node_size_factor,
            step=5,
            width=300
        )
        self.controls['node_size'].on_change('value', self._update_node_sizes)
        
        # Edge width
        self.controls['edge_width'] = Slider(
            title="Edge Width",
            start=1,
            end=10,
            value=self.edge_width_factor,
            step=1,
            width=300
        )
        self.controls['edge_width'].on_change('value', self._update_edge_widths)
        
        # Arrow size
        self.controls['arrow_size'] = Slider(
            title="Arrow Size",
            start=0.05,
            end=0.3,
            value=0.1,
            step=0.05,
            width=300
        )
        self.controls['arrow_size'].on_change('value', self._update_arrow_size)
        
        # Highlight mode
        self.controls['highlight_mode'] = RadioGroup(
            labels=["None", "Neighbors", "Components", "Paths"],
            active=0,
            width=300
        )
        self.controls['highlight_mode'].on_change('active', self._update_highlight_mode)
        
        # Show/Hide options
        self.controls['show_options'] = CheckboxGroup(
            labels=["Show Labels", "Show Edges", "Show Hyperedges", "Show Arrows"],
            active=[0, 1, 2, 3],  # All visible by default
            width=300
        )
        self.controls['show_options'].on_change('active', self._update_visibility)
        
        # Layout
        return column(
            Div(text="<h3>Visualization Controls</h3>"),
            self.controls['layout'],
            self.controls['color_scheme'],
            self.controls['node_size'],
            self.controls['edge_width'],
            self.controls['arrow_size'],
            Div(text="<b>Highlight Mode:</b>"),
            self.controls['highlight_mode'],
            Div(text="<b>Display Options:</b>"),
            self.controls['show_options']
        )
    
    def create_statistics_panel(self) -> layout:
        """
        Create statistics and information panel.
        
        Returns:
            Bokeh layout with statistics displays
        """
        # Overall statistics
        self.info_displays['overall_stats'] = Div(
            text=self._get_overall_statistics(),
            width=300,
            height=200
        )
        
        # Selected node info
        self.info_displays['selection_info'] = Div(
            text="<b>Selection:</b> None",
            width=300,
            height=100
        )
        
        # Evolution history
        self.info_displays['history_info'] = Div(
            text=self._get_history_info(),
            width=300,
            height=150
        )
        
        # Performance metrics
        self.info_displays['performance'] = Div(
            text=self._get_performance_info(),
            width=300,
            height=100
        )
        
        # Layout
        return column(
            Div(text="<h3>Statistics</h3>"),
            self.info_displays['overall_stats'],
            self.info_displays['selection_info'],
            self.info_displays['history_info'],
            self.info_displays['performance']
        )
    
    def create_multi_view_dashboard(self) -> layout:
        """
        Create a comprehensive multi-view dashboard.
        
        Returns:
            Complete dashboard layout
        """
        # Main visualization
        main_plot = self.create_main_plot(width=800, height=600)
        
        # Control panels as tabs
        evolution_tab = Panel(
            child=self.create_evolution_controls(),
            title="Evolution"
        )
        
        rules_tab = Panel(
            child=self.create_rule_controls(),
            title="Rules"
        )
        
        visual_tab = Panel(
            child=self.create_visualization_controls(),
            title="Visualization"
        )
        
        stats_tab = Panel(
            child=self.create_statistics_panel(),
            title="Statistics"
        )
        
        control_tabs = Tabs(tabs=[evolution_tab, rules_tab, visual_tab, stats_tab])
        
        # Additional views
        additional_views = self._create_additional_views()
        
        # Export controls
        export_controls = self._create_export_controls()
        
        # Complete layout
        dashboard = layout([
            [main_plot, control_tabs],
            [additional_views],
            [export_controls]
        ])
        
        return dashboard
    
    def _create_additional_views(self) -> layout:
        """Create additional visualization views."""
        # Adjacency matrix view
        adj_plot = self._create_adjacency_plot()
        
        # Degree distribution
        degree_plot = self._create_degree_distribution_plot()
        
        # Evolution timeline
        timeline_plot = self._create_evolution_timeline()
        
        return row(adj_plot, degree_plot, timeline_plot)
    
    def _create_adjacency_plot(self) -> figure:
        """Create adjacency matrix visualization."""
        p = figure(
            title="Adjacency Matrix",
            width=250,
            height=250,
            toolbar_location=None,
            tools=""
        )
        
        # Will be updated dynamically
        self.controls['adj_plot'] = p
        return p
    
    def _create_degree_distribution_plot(self) -> figure:
        """Create degree distribution plot."""
        p = figure(
            title="Degree Distribution",
            width=250,
            height=250,
            toolbar_location=None,
            tools=""
        )
        
        p.vbar(x=[1, 2, 3], top=[3, 2, 1], width=0.8)
        p.xaxis.axis_label = "Degree"
        p.yaxis.axis_label = "Count"
        
        self.controls['degree_plot'] = p
        return p
    
    def _create_evolution_timeline(self) -> figure:
        """Create evolution timeline visualization."""
        p = figure(
            title="Evolution Timeline",
            width=250,
            height=250,
            toolbar_location=None,
            tools="",
            x_axis_type="linear"
        )
        
        p.line(x=[0, 1, 2], y=[4, 6, 8], line_width=2)
        p.xaxis.axis_label = "Step"
        p.yaxis.axis_label = "Nodes"
        
        self.controls['timeline_plot'] = p
        return p
    
    def _create_export_controls(self) -> layout:
        """Create export and save controls."""
        # Export format
        export_format = Select(
            title="Export Format:",
            value="json",
            options=[("json", "JSON"), ("graphml", "GraphML"), ("png", "PNG")],
            width=150
        )
        
        # Export button
        export_btn = Button(
            label="Export",
            button_type="primary",
            width=100
        )
        export_btn.on_click(partial(self._export_data, export_format))
        
        # Save session
        save_btn = Button(
            label="Save Session",
            button_type="success",
            width=120
        )
        save_btn.on_click(self._save_session)
        
        # Load session
        load_btn = Button(
            label="Load Session",
            button_type="default",
            width=120
        )
        load_btn.on_click(self._load_session)
        
        return row(
            export_format,
            export_btn,
            save_btn,
            load_btn
        )
    
    def _update_data_sources(self) -> None:
        """Update all data sources with current hypergraph state."""
        if not self.processor.nodes:
            self.node_source = ColumnDataSource(data=dict(
                x=[], y=[], label=[], size=[], color=[], alpha=[], 
                degree=[], neighbors=[]
            ))
            self.node_positions = {}
            return
        
        # Compute layout and store for consistent use
        basic_viz = BasicVisualizer(self.processor)
        pos = basic_viz.compute_layout(self.current_layout)
        self.node_positions = pos  # Store positions for edge drawing
        
        # Prepare node data
        node_data = {
            'x': [],
            'y': [],
            'label': [],
            'size': [],
            'color': [],
            'alpha': [],
            'degree': [],
            'neighbors': []
        }
        
        for node in self.processor.nodes:
            if node in pos:
                node_data['x'].append(pos[node][0])
                node_data['y'].append(pos[node][1])
                node_data['label'].append(str(node))
                node_data['degree'].append(self.processor.get_node_degree(node))
                node_data['neighbors'].append(str(list(self.processor.find_neighbors(node))))
                
                # Enhanced size based on degree with better scaling
                base_size = self.node_size_factor
                degree_bonus = self.processor.get_node_degree(node) * 4
                node_data['size'].append(base_size + degree_bonus)
                
                # Color based on scheme
                node_data['color'].append(self._get_node_color(node))
                
                # Enhanced alpha for selection with better contrast
                node_data['alpha'].append(
                    1.0 if node in self.selected_nodes else 0.85
                )
        
        if self.node_source is None:
            self.node_source = ColumnDataSource(data=node_data)
        else:
            self.node_source.data = node_data
    
    def _draw_edges(self) -> None:
        """Draw all edges and hyperedges."""
        # This method is now primarily for updating edge properties
        # The actual edge drawing happens in create_main_plot and _update_all_displays
        pass
    
    def _add_arrow(self, x1: float, y1: float, x2: float, y2: float, edge_id: str) -> None:
        """
        Add directional arrow to an edge.
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates  
            edge_id: Edge identifier for styling
        """
        if not self.plot:
            return
            
        import math
        
        # Calculate arrow parameters
        arrow_length = 0.1  # Default arrow length
        if 'arrow_size' in self.controls:
            arrow_length = self.controls['arrow_size'].value
        arrow_angle = 0.5   # Angle of arrow head in radians
        
        # Calculate the direction vector
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 0.001:  # Avoid division by zero
            return
        
        # Normalize direction vector
        dx_norm = dx / length
        dy_norm = dy / length
        
        # Calculate arrow head position (offset from node to avoid overlap)
        node_radius = 0.05  # Approximate node radius in plot coordinates
        arrow_start_x = x2 - dx_norm * node_radius
        arrow_start_y = y2 - dy_norm * node_radius
        
        # Calculate arrow head points
        # Left arrow line
        left_x = arrow_start_x - arrow_length * (dx_norm * math.cos(arrow_angle) - dy_norm * math.sin(arrow_angle))
        left_y = arrow_start_y - arrow_length * (dy_norm * math.cos(arrow_angle) + dx_norm * math.sin(arrow_angle))
        
        # Right arrow line  
        right_x = arrow_start_x - arrow_length * (dx_norm * math.cos(-arrow_angle) - dy_norm * math.sin(-arrow_angle))
        right_y = arrow_start_y - arrow_length * (dy_norm * math.cos(-arrow_angle) + dx_norm * math.sin(-arrow_angle))
        
        # Enhanced arrow styling
        arrow_color = "#2C3E50"  # Dark slate color
        
        # Draw arrow head lines with enhanced styling
        self.plot.line(
            [arrow_start_x, left_x], [arrow_start_y, left_y],
            line_width=self.edge_width_factor + 1,
            line_color=arrow_color,
            alpha=self.arrow_alpha,
            line_cap="round"
        )
        
        self.plot.line(
            [arrow_start_x, right_x], [arrow_start_y, right_y],
            line_width=self.edge_width_factor + 1,
            line_color=arrow_color,
            alpha=self.arrow_alpha,
            line_cap="round"
        )
        
        # Enhanced triangular arrow head with gradient effect
        triangle_x = [arrow_start_x, left_x, right_x, arrow_start_x]
        triangle_y = [arrow_start_y, left_y, right_y, arrow_start_y]
        
        self.plot.patch(
            triangle_x, triangle_y,
            color=arrow_color,
            alpha=0.8,
            line_color=arrow_color,
            line_width=2,
            line_alpha=0.9
        )

    def _get_node_color(self, node: Any) -> str:
        """Get node color based on current color scheme."""
        if self.color_scheme == 'mathematica':
            # Enhanced Mathematica-style colors with better saturation
            mathematica_colors = [
                "#4472C4",  # Deep Blue
                "#E67C00",  # Rich Orange  
                "#70AD47",  # Forest Green
                "#FF4444",  # Bright Red
                "#9966CC",  # Royal Purple
                "#D2691E",  # Chocolate
                "#FF69B4",  # Hot Pink
                "#708090",  # Slate Gray
                "#32CD32",  # Lime Green
                "#FF6347"   # Tomato
            ]
            return mathematica_colors[hash(str(node)) % len(mathematica_colors)]
        elif self.color_scheme == 'degree':
            degree = self.processor.get_node_degree(node)
            colors = Viridis[11]
            return colors[min(degree, 10)]
        elif self.color_scheme == 'component':
            # Color by connected component
            return Category20[20][hash(str(node)) % 20]
        elif self.color_scheme == 'evolution':
            # Color by when node was created
            return Spectral[11][self.current_history_index % 11]
        elif self.color_scheme == 'viridis':
            # Viridis color scheme
            degree = self.processor.get_node_degree(node)
            return Viridis[11][min(degree, 10)]
        elif self.color_scheme == 'plasma':
            # Plasma color scheme 
            degree = self.processor.get_node_degree(node)
            from bokeh.palettes import Plasma
            return Plasma[11][min(degree, 10)]
        elif self.color_scheme == 'sunset':
            # Sunset color palette
            sunset_colors = [
                "#FF6B35", "#F7931E", "#FFD23F", "#F65058", "#FF8C42",
                "#E67E22", "#D63031", "#A0392A", "#744C28", "#8B4513"
            ]
            return sunset_colors[hash(str(node)) % len(sunset_colors)]
        elif self.color_scheme == 'ocean':
            # Ocean blue palette
            ocean_colors = [
                "#006994", "#13778C", "#3282A3", "#4A90A4", "#5DA3A8",
                "#7BB3AC", "#86C5B1", "#A7D8DE", "#B8E6E6", "#CAF0F8"
            ]
            return ocean_colors[hash(str(node)) % len(ocean_colors)]
        else:
            return "#4472C4"  # Default enhanced blue
    
    def _get_enhanced_edge_color(self, edge_id: str, node1: str, node2: str) -> str:
        """Get enhanced edge color based on connected nodes."""
        if self.edge_gradient:
            # Create gradient-like effect by blending node colors
            color1 = self._get_node_color(node1)
            color2 = self._get_node_color(node2)
            # For now, return a blend between the colors (simplified)
            if color1 == color2:
                return color1
            else:
                return "#7F8C8D"  # Neutral gray for different colored nodes
        else:
            return "#34495E"  # Default dark edge color
    
    def _get_evolution_info(self) -> str:
        """Get current evolution information."""
        return f"""
        <b>Evolution Status:</b> {'Running' if self.is_evolving else 'Stopped'}<br>
        <b>Current Step:</b> {self.processor.current_step}<br>
        <b>History Length:</b> {len(self.evolution_history)}<br>
        <b>Total Applications:</b> {self.rule_engine.application_count}
        """
    
    def _get_rule_info(self) -> str:
        """Get information about current rules."""
        if not self.rule_engine.rules:
            return "<b>No rules loaded</b>"
        
        rule_list = "<b>Active Rules:</b><br>"
        for rule in self.rule_engine.rules[:5]:  # Show first 5
            rule_list += f"• {rule.name} (priority: {rule.priority})<br>"
        
        if len(self.rule_engine.rules) > 5:
            rule_list += f"• ... and {len(self.rule_engine.rules) - 5} more<br>"
        
        return rule_list
    
    def _get_overall_statistics(self) -> str:
        """Get overall hypergraph statistics."""
        stats = self.processor.compute_statistics()
        
        return f"""
        <b>Hypergraph Statistics:</b><br>
        <b>Nodes:</b> {stats['node_count']}<br>
        <b>Edges:</b> {stats['edge_count']}<br>
        <b>Components:</b> {stats['connected_components']}<br>
        <b>Clustering:</b> {stats.get('clustering_coefficient', 0):.3f}<br>
        <b>Avg Degree:</b> {stats.get('average_degree', 0):.2f}<br>
        <b>Max Degree:</b> {stats.get('max_degree', 0)}<br>
        """
    
    def _get_history_info(self) -> str:
        """Get evolution history information."""
        if not self.evolution_history:
            return "<b>No evolution history</b>"
        
        return f"""
        <b>Evolution History:</b><br>
        <b>Total Steps:</b> {len(self.evolution_history)}<br>
        <b>Current Position:</b> {self.current_history_index + 1}<br>
        <b>First State:</b> {self.evolution_history[0]['node_count']} nodes<br>
        <b>Current State:</b> {self.processor.node_count} nodes<br>
        <b>Growth Rate:</b> {(self.processor.node_count / max(self.evolution_history[0]['node_count'], 1)):.2f}x
        """
    
    def _get_performance_info(self) -> str:
        """Get performance metrics."""
        return f"""
        <b>Performance:</b><br>
        <b>Update Rate:</b> {1000/self.animation_speed:.1f} Hz<br>
        <b>Nodes Rendered:</b> {self.processor.node_count}<br>
        <b>Edges Rendered:</b> {self.processor.edge_count}
        """
    
    # Event Handlers
    def _toggle_evolution(self) -> None:
        """Toggle evolution play/pause."""
        if self.is_evolving:
            self.is_evolving = False
            self.controls['play_pause'].label = "▶ Play"
            self.controls['play_pause'].button_type = "success"
        else:
            self.is_evolving = True
            self.controls['play_pause'].label = "⏸ Pause"
            self.controls['play_pause'].button_type = "warning"
            curdoc().add_next_tick_callback(self._evolution_step)
    
    def _evolution_step(self) -> None:
        """Perform one evolution step."""
        if not self.is_evolving:
            return
        
        # Apply evolution
        max_apps = int(self.controls['max_applications'].value)
        self.processor = self.rule_engine.apply_single_step(
            self.processor, max_applications=max_apps
        )
        
        # Save to history
        self.evolution_history.append(self.processor.snapshot())
        self.current_history_index = len(self.evolution_history) - 1
        
        # Update visualization
        self._update_all_displays()
        
        # Schedule next step
        if self.is_evolving:
            curdoc().add_timeout_callback(
                self._evolution_step, self.animation_speed
            )
    
    def _step_forward(self) -> None:
        """Step forward in evolution."""
        if self.current_history_index < len(self.evolution_history) - 1:
            # Move forward in history
            self.current_history_index += 1
            snapshot = self.evolution_history[self.current_history_index]
            self.processor.load_from_snapshot(snapshot)
        else:
            # Evolve one step
            max_apps = int(self.controls['max_applications'].value)
            self.processor = self.rule_engine.apply_single_step(
                self.processor, max_applications=max_apps
            )
            self.evolution_history.append(self.processor.snapshot())
            self.current_history_index = len(self.evolution_history) - 1
        
        self._update_all_displays()
    
    def _step_backward(self) -> None:
        """Step backward in evolution history."""
        if self.current_history_index > 0:
            self.current_history_index -= 1
            snapshot = self.evolution_history[self.current_history_index]
            self.processor.load_from_snapshot(snapshot)
            self._update_all_displays()
    
    def _reset_evolution(self) -> None:
        """Reset to initial state."""
        if self.evolution_history:
            self.processor.load_from_snapshot(self.evolution_history[0])
            self.current_history_index = 0
        else:
            self.processor = HypergraphProcessor()
        
        self.rule_engine.reset_statistics()
        self._update_all_displays()
    
    def _update_speed(self, attr, old, new) -> None:
        """Update animation speed."""
        self.animation_speed = new
    
    def _update_layout(self, attr, old, new) -> None:
        """Update layout algorithm."""
        self.current_layout = new
        self._update_all_displays()
    
    def _update_colors(self, attr, old, new) -> None:
        """Update color scheme."""
        self.color_scheme = new
        self._update_all_displays()
    
    def _update_node_sizes(self, attr, old, new) -> None:
        """Update node sizes."""
        self.node_size_factor = new
        self._update_all_displays()
    
    def _update_edge_widths(self, attr, old, new) -> None:
        """Update edge widths."""
        self.edge_width_factor = new
        self._update_all_displays()
    
    def _update_arrow_size(self, attr, old, new) -> None:
        """Update arrow size."""
        # Arrow size is used directly in _add_arrow method
        self._update_all_displays()
    
    def _update_highlight_mode(self, attr, old, new) -> None:
        """Update highlight mode."""
        modes = ["none", "neighbors", "components", "paths"]
        self.highlight_mode = modes[new]
        self._update_highlights()
    
    def _update_visibility(self, attr, old, new) -> None:
        """Update element visibility."""
        # Update based on checkbox selections
        self._update_all_displays()
    
    def _on_tap(self, event) -> None:
        """Handle tap events for node selection."""
        # Find closest node to tap location
        if self.node_source and len(self.node_source.data['x']) > 0:
            x_coords = np.array(self.node_source.data['x'])
            y_coords = np.array(self.node_source.data['y'])
            
            distances = np.sqrt((x_coords - event.x)**2 + (y_coords - event.y)**2)
            closest_idx = np.argmin(distances)
            
            if distances[closest_idx] < 0.1:  # Threshold for selection
                node_label = self.node_source.data['label'][closest_idx]
                
                if node_label in self.selected_nodes:
                    self.selected_nodes.remove(node_label)
                else:
                    self.selected_nodes.add(node_label)
                
                self._update_selection_display()
                self._update_highlights()
    
    def _update_selection_display(self) -> None:
        """Update selection information display."""
        if not self.selected_nodes:
            self.info_displays['selection_info'].text = "<b>Selection:</b> None"
        else:
            info = f"<b>Selected Nodes:</b> {', '.join(self.selected_nodes)}<br>"
            
            # Add detailed info for single selection
            if len(self.selected_nodes) == 1:
                node = list(self.selected_nodes)[0]
                degree = self.processor.get_node_degree(node)
                neighbors = self.processor.find_neighbors(node)
                info += f"<b>Degree:</b> {degree}<br>"
                info += f"<b>Neighbors:</b> {', '.join(map(str, neighbors))}"
            
            self.info_displays['selection_info'].text = info
    
    def _update_highlights(self) -> None:
        """Update node highlighting based on mode."""
        if self.highlight_mode == 'neighbors' and self.selected_nodes:
            # Highlight neighbors of selected nodes
            highlighted = set()
            for node in self.selected_nodes:
                highlighted.update(self.processor.find_neighbors(node))
            
            # Update alpha values
            if self.node_source:
                alphas = []
                for label in self.node_source.data['label']:
                    if label in self.selected_nodes:
                        alphas.append(1.0)
                    elif label in highlighted:
                        alphas.append(0.9)
                    else:
                        alphas.append(0.3)
                
                self.node_source.data['alpha'] = alphas
        else:
            # Reset to normal
            if self.node_source:
                self.node_source.data['alpha'] = [0.8] * len(self.node_source.data['label'])
    
    def _update_all_displays(self) -> None:
        """Update all visualization displays."""
        # Update node data
        self._update_data_sources()
        
        # Clear and redraw the plot if it exists
        if self.plot:
            # Clear all renderers
            self.plot.renderers = []
            
            # Redraw edges using the same positions as nodes
            if self.processor.edges and hasattr(self, 'node_positions'):
                pos = self.node_positions
                
                # Check if edges should be shown
                show_edges = True
                if 'show_options' in self.controls and hasattr(self.controls['show_options'], 'active'):
                    show_edges = 1 in self.controls['show_options'].active
                
                if show_edges:
                    for edge_id, edge_nodes in self.processor.edges.items():
                        if len(edge_nodes) == 2 and edge_nodes[0] in pos and edge_nodes[1] in pos:
                            x1, y1 = pos[edge_nodes[0]]
                            x2, y2 = pos[edge_nodes[1]]
                            
                            # Draw line
                            self.plot.line(
                                [x1, x2], [y1, y2],
                                line_width=self.edge_width_factor,
                                line_color="gray",
                                alpha=0.6
                            )
                            
                            # Add directional arrow if enabled
                            show_arrows = True
                            if 'show_options' in self.controls and hasattr(self.controls['show_options'], 'active'):
                                show_arrows = 3 in self.controls['show_options'].active
                            
                            if show_arrows:
                                self._add_arrow(x1, y1, x2, y2, edge_id)
            
            # Redraw nodes
            if self.node_source:
                node_renderer = self.plot.scatter(
                    'x', 'y', size='size', color='color', alpha='alpha',
                    line_color="black", line_width=2, source=self.node_source,
                    selection_color="red", selection_alpha=1.0,
                    nonselection_alpha=0.6
                )
                
                # Add hover tool for nodes
                hover = HoverTool(
                    tooltips=[
                        ("Node", "@label"),
                        ("Degree", "@degree"),
                        ("Neighbors", "@neighbors"),
                        ("Position", "(@x, @y)")
                    ],
                    renderers=[node_renderer]
                )
                self.plot.add_tools(hover)
                
                # Add node labels
                show_labels = True
                if 'show_options' in self.controls and hasattr(self.controls['show_options'], 'active'):
                    show_labels = 0 in self.controls['show_options'].active
                
                if show_labels:
                    self.plot.text(
                        'x', 'y', text='label', text_align="center",
                        text_baseline="middle", text_font_size="10pt",
                        source=self.node_source
                    )
        
        # Update info displays if they exist
        if 'evolution_info' in self.info_displays:
            self.info_displays['evolution_info'].text = self._get_evolution_info()
        if 'overall_stats' in self.info_displays:
            self.info_displays['overall_stats'].text = self._get_overall_statistics()
        if 'history_info' in self.info_displays:
            self.info_displays['history_info'].text = self._get_history_info()
        if 'performance' in self.info_displays:
            self.info_displays['performance'].text = self._get_performance_info()
    
    def _show_add_rule_dialog(self) -> None:
        """Show dialog for adding new rule."""
        # In a real implementation, this would show a modal dialog
        # For now, add a simple predefined rule
        new_rule = RewriteRule(
            name=f"custom_rule_{len(self.rule_engine.rules)}",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"]), ("e2", ["B", "C"])],
            priority=int(self.controls['rule_priority'].value)
        )
        self.rule_engine.add_rule(new_rule)
        
        # Update rule selector
        rule_options = [(rule.name, rule.name) for rule in self.rule_engine.rules]
        self.controls['rule_select'].options = rule_options
        self.info_displays['rule_info'].text = self._get_rule_info()
    
    def _remove_selected_rule(self) -> None:
        """Remove currently selected rule."""
        selected_rule = self.controls['rule_select'].value
        if selected_rule:
            self.rule_engine.remove_rule(selected_rule)
            
            # Update rule selector
            rule_options = [(rule.name, rule.name) for rule in self.rule_engine.rules]
            self.controls['rule_select'].options = rule_options
            if rule_options:
                self.controls['rule_select'].value = rule_options[0][0]
            
            self.info_displays['rule_info'].text = self._get_rule_info()
    
    def _load_rule_preset(self, attr, old, new) -> None:
        """Load a predefined rule set."""
        if new == 'basic':
            rules = WolframRuleEngine.create_basic_rules()
        elif new == 'advanced':
            rules = WolframRuleEngine.create_advanced_rules()
        else:
            return
        
        # Clear existing rules and add new ones
        self.rule_engine.rules = []
        for rule in rules:
            self.rule_engine.add_rule(rule)
        
        # Update UI
        rule_options = [(rule.name, rule.name) for rule in self.rule_engine.rules]
        self.controls['rule_select'].options = rule_options
        if rule_options:
            self.controls['rule_select'].value = rule_options[0][0]
        
        self.info_displays['rule_info'].text = self._get_rule_info()
    
    def _export_data(self, format_select, event) -> None:
        """Export current hypergraph data."""
        export_format = format_select.value
        
        if export_format == 'json':
            # Export as JSON
            data = {
                'snapshot': self.processor.snapshot(),
                'rules': [
                    {
                        'name': rule.name,
                        'pattern': rule.pattern,
                        'replacement': rule.replacement,
                        'priority': rule.priority
                    }
                    for rule in self.rule_engine.rules
                ],
                'history': self.evolution_history,
                'timestamp': datetime.now().isoformat()
            }
            
            # In a real app, this would trigger a download
            print(f"Export data: {json.dumps(data, indent=2)[:200]}...")
    
    def _save_session(self, event) -> None:
        """Save current session state."""
        session_data = {
            'processor_snapshot': self.processor.snapshot(),
            'evolution_history': self.evolution_history,
            'current_index': self.current_history_index,
            'rules': [(r.name, r.pattern, r.replacement, r.priority) 
                     for r in self.rule_engine.rules],
            'visualization_settings': {
                'layout': self.current_layout,
                'color_scheme': self.color_scheme,
                'node_size': self.node_size_factor,
                'edge_width': self.edge_width_factor
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to data manager
        data_manager = DataManager("./sessions", backend='json')
        session_id = data_manager.save_experiment(
            f"interactive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            self.processor,
            description="Interactive visualization session",
            parameters=session_data
        )
        
        print(f"Session saved: {session_id}")
    
    def _load_session(self, event) -> None:
        """Load a saved session."""
        # In a real implementation, this would show a file dialog
        print("Load session functionality would be implemented here")
    
    def serve_app(self, port: int = 5006) -> None:
        """
        Serve the interactive visualization as a Bokeh app.
        
        Args:
            port: Port to serve on
        """
        def modify_doc(doc):
            doc.add_root(self.create_multi_view_dashboard())
            doc.title = "Wolfram Physics Interactive Visualizer"
        
        from bokeh.server.server import Server
        server = Server({'/': modify_doc}, port=port)
        server.start()
        
        print(f"Interactive visualizer running at http://localhost:{port}")
        server.io_loop.start()
    
    def __str__(self) -> str:
        """String representation."""
        return f"InteractiveVisualizer(nodes={self.processor.node_count}, rules={len(self.rule_engine.rules)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()