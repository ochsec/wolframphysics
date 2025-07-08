#!/usr/bin/env python
"""
Bokeh server application for Wolfram Physics Interactive Visualizer.
Run with: bokeh serve --show bokeh_app.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from bokeh.io import curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import Div, Button, Select
from bokeh.plotting import figure

from src.wolfram_physics.hypergraph_processor import HypergraphProcessor
from src.wolfram_physics.rule_engine import WolframRuleEngine
from src.wolfram_physics.interactive_visualizer import InteractiveVisualizer


def create_app():
    """Create the main Bokeh application."""
    
    # Header
    header = Div(
        text="""
        <h1>Wolfram Physics Interactive Visualizer</h1>
        <p>Real-time hypergraph evolution with interactive controls and multi-view visualization.</p>
        """,
        width=1200,
        height=80
    )
    
    # Initialize with example hypergraph
    initial_edges = {
        'e1': ['A', 'B'],
        'e2': ['B', 'C'],
        'e3': ['C', 'D'],
        'e4': ['D', 'A'],
        'e5': ['A', 'C']
    }
    
    processor = HypergraphProcessor(initial_edges)
    
    # Initialize rule engine with basic rules
    engine = WolframRuleEngine()
    basic_rules = WolframRuleEngine.create_basic_rules()
    for rule in basic_rules:
        engine.add_rule(rule)
    
    # Create interactive visualizer
    visualizer = InteractiveVisualizer(processor, engine)
    
    # Add initial snapshot to history
    visualizer.evolution_history.append(processor.snapshot())
    
    # Create the dashboard
    dashboard = visualizer.create_multi_view_dashboard()
    
    # Preset examples
    example_selector = Select(
        title="Load Example:",
        value="custom",
        options=[
            ("custom", "Custom"),
            ("simple", "Simple Triangle"),
            ("complex", "Complex Network"),
            ("lattice", "Lattice Structure"),
            ("random", "Random Graph")
        ],
        width=200
    )
    
    def load_example(attr, old, new):
        """Load a preset example."""
        nonlocal processor, visualizer
        
        if new == "simple":
            edges = {
                'e1': ['A', 'B'],
                'e2': ['B', 'C'],
                'e3': ['C', 'A']
            }
        elif new == "complex":
            edges = {
                'e1': ['A', 'B'],
                'e2': ['B', 'C'],
                'e3': ['C', 'D'],
                'e4': ['D', 'E'],
                'e5': ['E', 'A'],
                'e6': ['A', 'C', 'E'],
                'e7': ['B', 'D', 'F'],
                'e8': ['F', 'G'],
                'e9': ['G', 'H'],
                'e10': ['H', 'A']
            }
        elif new == "lattice":
            # Create a small lattice
            edges = {}
            nodes_per_row = 4
            for i in range(nodes_per_row):
                for j in range(nodes_per_row):
                    node = f"{i},{j}"
                    # Horizontal edge
                    if j < nodes_per_row - 1:
                        edges[f"h_{i}_{j}"] = [node, f"{i},{j+1}"]
                    # Vertical edge
                    if i < nodes_per_row - 1:
                        edges[f"v_{i}_{j}"] = [node, f"{i+1},{j}"]
        elif new == "random":
            import random
            edges = {}
            nodes = [f"N{i}" for i in range(10)]
            for i in range(15):
                n1, n2 = random.sample(nodes, 2)
                edges[f"e{i}"] = [n1, n2]
        else:
            return
        
        # Update processor
        visualizer.processor = HypergraphProcessor(edges)
        visualizer.evolution_history = [visualizer.processor.snapshot()]
        visualizer.current_history_index = 0
        # Update displays after controls are initialized
        if hasattr(visualizer, '_update_all_displays'):
            visualizer._update_all_displays()
    
    example_selector.on_change('value', load_example)
    
    # Help button
    help_button = Button(label="Help", button_type="default", width=80)
    
    def show_help():
        help_text = """
        <h3>Quick Start Guide</h3>
        <ul>
            <li><b>Evolution:</b> Use Play/Pause to run automatic evolution</li>
            <li><b>Step Control:</b> Use Step Forward/Back to control evolution manually</li>
            <li><b>Rules:</b> Select and modify transformation rules</li>
            <li><b>Visualization:</b> Change layout, colors, and display options</li>
            <li><b>Selection:</b> Click nodes to select and see details</li>
            <li><b>Export:</b> Save your work in various formats</li>
        </ul>
        """
        visualizer.info_displays['overall_stats'].text = help_text
    
    help_button.on_click(show_help)
    
    # Top controls
    top_controls = row(example_selector, help_button)
    
    # Complete layout
    app_layout = column(
        header,
        top_controls,
        dashboard
    )
    
    return app_layout


# Create the document
def modify_doc(doc):
    """Modify the Bokeh document."""
    doc.title = "Wolfram Physics Interactive Visualizer"
    doc.add_root(create_app())
    
    # Add periodic callback for evolution updates
    # This is handled internally by the InteractiveVisualizer


# This is important for Bokeh server
curdoc().title = "Wolfram Physics Interactive Visualizer"
curdoc().add_root(create_app())