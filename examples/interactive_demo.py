"""
Interactive demonstration of Phase 2 features for Wolfram Physics Project.
Shows real-time visualization, 3D views, and interactive controls.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wolfram_physics.hypergraph_processor import HypergraphProcessor
from src.wolfram_physics.rule_engine import WolframRuleEngine, RewriteRule
from src.wolfram_physics.interactive_visualizer import InteractiveVisualizer
from src.wolfram_physics.visualizer_3d import Visualizer3D
import plotly.graph_objects as go


def demonstrate_3d_visualization():
    """Demonstrate 3D visualization capabilities."""
    print("\n" + "="*50)
    print("3D VISUALIZATION DEMONSTRATION")
    print("="*50)
    
    # Create a complex hypergraph
    edges = {
        'e1': ['A', 'B'],
        'e2': ['B', 'C'],
        'e3': ['C', 'D'],
        'e4': ['D', 'E'],
        'e5': ['E', 'A'],
        'e6': ['A', 'C', 'E'],  # Hyperedge
        'e7': ['B', 'D', 'F'],  # Hyperedge
        'e8': ['F', 'G'],
        'e9': ['G', 'H'],
        'e10': ['H', 'A']
    }
    
    processor = HypergraphProcessor(edges)
    print(f"Created hypergraph: {processor}")
    
    # Create 3D visualizer
    viz_3d = Visualizer3D(processor)
    
    # Create single 3D plot
    print("\nGenerating 3D visualization...")
    fig = viz_3d.create_3d_plot(
        title="3D Wolfram Physics Hypergraph",
        layout='spring_3d',
        show_hyperedges=True,
        node_size=10
    )
    
    # Save as HTML
    output_file = "examples/hypergraph_3d.html"
    viz_3d.export_3d_html(output_file)
    print(f"3D visualization saved as '{output_file}'")
    
    # Create multi-view comparison
    print("\nGenerating multi-view 3D comparison...")
    multi_fig = viz_3d.create_multi_view_3d()
    multi_fig.write_html("examples/hypergraph_3d_multiview.html")
    print("Multi-view comparison saved as 'examples/hypergraph_3d_multiview.html'")
    
    return processor, viz_3d


def demonstrate_evolution_animation_3d():
    """Demonstrate 3D evolution animation."""
    print("\n" + "="*50)
    print("3D EVOLUTION ANIMATION DEMONSTRATION")
    print("="*50)
    
    # Create initial hypergraph
    processor = HypergraphProcessor({'e1': ['A', 'B'], 'e2': ['B', 'C']})
    
    # Create rule engine
    engine = WolframRuleEngine()
    
    # Add expansion rule
    expansion_rule = RewriteRule(
        name="3d_expansion",
        pattern=[("e1", ["A", "B"])],
        replacement=[("e1", ["A", "D"]), ("e2", ["D", "B"]), ("e3", ["A", "B", "D"])]
    )
    engine.add_rule(expansion_rule)
    
    # Evolve and collect history
    evolution_history = [processor.snapshot()]
    
    print("Evolving hypergraph...")
    for i in range(5):
        processor = engine.apply_single_step(processor)
        evolution_history.append(processor.snapshot())
        print(f"  Step {i+1}: {processor.node_count} nodes, {processor.edge_count} edges")
    
    # Create 3D animation
    print("\nGenerating 3D evolution animation...")
    viz_3d = Visualizer3D(processor)
    
    anim_fig = viz_3d.create_evolution_animation_3d(
        evolution_history,
        layout='spring_3d',
        duration=1000
    )
    
    anim_fig.write_html("examples/evolution_3d_animation.html")
    print("3D animation saved as 'examples/evolution_3d_animation.html'")
    
    return evolution_history


def demonstrate_interactive_features():
    """Demonstrate interactive visualization features."""
    print("\n" + "="*50)
    print("INTERACTIVE FEATURES DEMONSTRATION")
    print("="*50)
    
    # Create processor and engine
    processor = HypergraphProcessor({
        'e1': ['A', 'B'],
        'e2': ['B', 'C'],
        'e3': ['C', 'D']
    })
    
    engine = WolframRuleEngine()
    
    # Add various rules
    rules = [
        RewriteRule(
            name="node_split",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "X"]), ("e2", ["X", "B"])],
            priority=2
        ),
        RewriteRule(
            name="edge_merge",
            pattern=[("e1", ["A", "B"]), ("e2", ["B", "C"])],
            replacement=[("e1", ["A", "B", "C"])],
            priority=1
        ),
        RewriteRule(
            name="triangle_close",
            pattern=[("e1", ["A", "B"]), ("e2", ["B", "C"])],
            replacement=[("e1", ["A", "B"]), ("e2", ["B", "C"]), ("e3", ["C", "A"])],
            priority=3
        )
    ]
    
    for rule in rules:
        engine.add_rule(rule)
    
    print(f"Loaded {len(engine.rules)} rules")
    
    # Create interactive visualizer
    visualizer = InteractiveVisualizer(processor, engine)
    
    print("\nInteractive features available:")
    print("  - Real-time evolution with play/pause/step controls")
    print("  - Dynamic rule management and editing")
    print("  - Multiple layout algorithms (spring, circular, hierarchical)")
    print("  - Color schemes (by degree, component, evolution)")
    print("  - Node selection and highlighting")
    print("  - Multi-view dashboard with statistics")
    print("  - Export capabilities (JSON, GraphML, PNG)")
    
    print("\nTo run the interactive server:")
    print("  bokeh serve --show bokeh_app.py")
    
    return visualizer


def create_example_gallery():
    """Create a gallery of example visualizations."""
    print("\n" + "="*50)
    print("CREATING EXAMPLE GALLERY")
    print("="*50)
    
    examples = {
        "Triangle Network": {
            'e1': ['A', 'B'],
            'e2': ['B', 'C'],
            'e3': ['C', 'A']
        },
        "Star Graph": {
            'e1': ['Center', 'A'],
            'e2': ['Center', 'B'],
            'e3': ['Center', 'C'],
            'e4': ['Center', 'D'],
            'e5': ['Center', 'E']
        },
        "Hyperedge Network": {
            'e1': ['A', 'B', 'C'],
            'e2': ['B', 'C', 'D'],
            'e3': ['C', 'D', 'E'],
            'e4': ['D', 'E', 'F'],
            'e5': ['E', 'F', 'A']
        },
        "Bipartite Graph": {
            'e1': ['U1', 'V1'],
            'e2': ['U1', 'V2'],
            'e3': ['U2', 'V1'],
            'e4': ['U2', 'V2'],
            'e5': ['U3', 'V1'],
            'e6': ['U3', 'V3']
        }
    }
    
    for name, edges in examples.items():
        print(f"\nCreating '{name}' example...")
        
        processor = HypergraphProcessor(edges)
        viz_3d = Visualizer3D(processor)
        
        # Create 3D visualization
        fig = viz_3d.create_3d_plot(
            title=f"Example: {name}",
            layout='spring_3d',
            show_hyperedges=True
        )
        
        filename = f"examples/gallery_{name.lower().replace(' ', '_')}.html"
        fig.write_html(filename)
        print(f"  Saved as '{filename}'")
    
    print("\nGallery created successfully!")


def demonstrate_performance_scaling():
    """Demonstrate performance with larger hypergraphs."""
    print("\n" + "="*50)
    print("PERFORMANCE SCALING DEMONSTRATION")
    print("="*50)
    
    import time
    
    sizes = [10, 50, 100, 200]
    
    for size in sizes:
        print(f"\nTesting with {size} nodes...")
        
        # Create random hypergraph
        edges = {}
        nodes = [f"N{i}" for i in range(size)]
        
        # Add regular edges
        for i in range(size * 2):
            import random
            n1, n2 = random.sample(nodes, 2)
            edges[f"e{i}"] = [n1, n2]
        
        # Add some hyperedges
        for i in range(size // 5):
            nodes_in_edge = random.sample(nodes, min(4, len(nodes)))
            edges[f"h{i}"] = nodes_in_edge
        
        processor = HypergraphProcessor(edges)
        
        # Time visualization creation
        start = time.time()
        viz_3d = Visualizer3D(processor)
        fig = viz_3d.create_3d_plot(show_hyperedges=True)
        creation_time = time.time() - start
        
        print(f"  Nodes: {processor.node_count}")
        print(f"  Edges: {processor.edge_count}")
        print(f"  3D visualization time: {creation_time:.3f}s")
        
        # Time layout computation
        start = time.time()
        viz_3d.compute_3d_layout('spring_3d')
        layout_time = time.time() - start
        print(f"  Layout computation time: {layout_time:.3f}s")


def main():
    """Main demonstration function."""
    print("WOLFRAM PHYSICS PROJECT - PHASE 2 DEMONSTRATION")
    print("Interactive Visualization and Enhanced UI")
    print("="*60)
    
    # Ensure examples directory exists
    os.makedirs("examples", exist_ok=True)
    
    try:
        # Run demonstrations
        processor, viz_3d = demonstrate_3d_visualization()
        evolution_history = demonstrate_evolution_animation_3d()
        visualizer = demonstrate_interactive_features()
        create_example_gallery()
        demonstrate_performance_scaling()
        
        print("\n" + "="*60)
        print("PHASE 2 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nGenerated files:")
        print("  - examples/hypergraph_3d.html")
        print("  - examples/hypergraph_3d_multiview.html")
        print("  - examples/evolution_3d_animation.html")
        print("  - examples/gallery_*.html")
        
        print("\nKey Phase 2 Features Implemented:")
        print("  ✓ Interactive Bokeh server application")
        print("  ✓ Real-time evolution controls")
        print("  ✓ Dynamic rule management")
        print("  ✓ 3D visualization with Plotly")
        print("  ✓ Multi-view dashboards")
        print("  ✓ Export capabilities")
        print("  ✓ Performance optimization")
        
        print("\nTo run the interactive server:")
        print("  bokeh serve --show bokeh_app.py")
        
        print("\nPhase 2 implementation is complete!")
        print("Ready for Phase 3: Distributed Computing and GPU Acceleration")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())