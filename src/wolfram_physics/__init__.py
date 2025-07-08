"""
Wolfram Physics Project - Python Framework for Hypergraph Evolution and Visualization

This package provides a comprehensive framework for exploring and visualizing
the Wolfram Physics Project's discrete computational models based on evolving hypergraphs.

Main Components:
- HypergraphProcessor: Core hypergraph processing and analysis
- WolframRuleEngine: Rule-based transformation system
- BasicVisualizer: Static and interactive visualization
- DataManager: Efficient storage and retrieval of evolution data
- InteractiveVisualizer: Real-time interactive visualization with Bokeh
- Visualizer3D: 3D visualization with Plotly

Phase 1 & 2 Implementation includes:
- Foundation hypergraph processing infrastructure
- Basic Wolfram rule system
- Static and interactive visualization frameworks
- Real-time evolution controls
- 3D visualization capabilities
- Data management and storage system
- Comprehensive test suite

Example Usage:
    >>> from wolfram_physics import HypergraphProcessor, WolframRuleEngine
    >>> from wolfram_physics import BasicVisualizer, DataManager
    >>> 
    >>> # Create initial hypergraph
    >>> processor = HypergraphProcessor({'e1': ['A', 'B'], 'e2': ['B', 'C']})
    >>> 
    >>> # Set up rule engine
    >>> engine = WolframRuleEngine()
    >>> engine.add_rule(RewriteRule(
    ...     name="expand",
    ...     pattern=[("e1", ["A", "B"])],
    ...     replacement=[("e1", ["A", "X"]), ("e2", ["X", "B"])]
    ... ))
    >>> 
    >>> # Evolve hypergraph
    >>> evolved = engine.evolve(processor, steps=5)
    >>> 
    >>> # Visualize results
    >>> visualizer = BasicVisualizer(evolved)
    >>> fig = visualizer.plot_static()
    >>> 
    >>> # Save experiment
    >>> data_manager = DataManager("./data")
    >>> experiment_id = data_manager.save_experiment("test", evolved)
"""

from .hypergraph_processor import HypergraphProcessor
from .rule_engine import WolframRuleEngine, RewriteRule
from .visualizer import BasicVisualizer
from .data_manager import DataManager
from .interactive_visualizer import InteractiveVisualizer
from .visualizer_3d import Visualizer3D

__version__ = "0.2.0"
__author__ = "Claude Code"
__email__ = "noreply@anthropic.com"
__description__ = "Python framework for Wolfram Physics Project exploration and visualization"

__all__ = [
    "HypergraphProcessor",
    "WolframRuleEngine", 
    "RewriteRule",
    "BasicVisualizer",
    "DataManager",
    "InteractiveVisualizer",
    "Visualizer3D"
]

# Package metadata
PACKAGE_INFO = {
    "name": "wolfram-physics",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/wolframphysics/python-framework",
    "license": "MIT",
    "keywords": ["wolfram", "physics", "hypergraph", "evolution", "visualization"],
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
}