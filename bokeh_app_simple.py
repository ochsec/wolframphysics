#!/usr/bin/env python
"""
Simple Bokeh app to test edge rendering.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.models import Div
from bokeh.plotting import figure

from src.wolfram_physics.hypergraph_processor import HypergraphProcessor
from src.wolfram_physics.rule_engine import WolframRuleEngine
from src.wolfram_physics.interactive_visualizer import InteractiveVisualizer


def create_simple_app():
    """Create a simple test app."""
    # Header
    header = Div(text="<h1>Simple Hypergraph Visualization</h1>")
    
    # Create triangle hypergraph
    processor = HypergraphProcessor({
        'e1': ['A', 'B'],
        'e2': ['B', 'C'],
        'e3': ['A', 'C']
    })
    
    engine = WolframRuleEngine()
    
    # Create visualizer
    visualizer = InteractiveVisualizer(processor, engine)
    
    # Create the main plot
    plot = visualizer.create_main_plot(width=600, height=600)
    
    # Info
    info = Div(text=f"""
        <p><b>Hypergraph Info:</b><br>
        Nodes: {list(processor.nodes)}<br>
        Edges: {dict(processor.edges)}<br>
        <br>
        <b>Note:</b> Arrows show direction from first to second node in each edge.<br>
        Edge directions: A→B, B→C, A→C<br>
        </p>
    """)
    
    return column(header, plot, info)


# Create document
curdoc().title = "Simple Hypergraph Test"
curdoc().add_root(create_simple_app())