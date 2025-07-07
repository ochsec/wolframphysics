"""
Basic demonstration of the Wolfram Physics Project framework.
This script shows how to use the core components to create and evolve hypergraphs.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for compatibility

from src.wolfram_physics.hypergraph_processor import HypergraphProcessor
from src.wolfram_physics.rule_engine import WolframRuleEngine, RewriteRule
from src.wolfram_physics.visualizer import BasicVisualizer
from src.wolfram_physics.data_manager import DataManager
import matplotlib.pyplot as plt


def create_simple_hypergraph():
    """Create a simple hypergraph for demonstration."""
    print("Creating initial hypergraph...")
    
    # Define initial hypergraph structure
    initial_edges = {
        'e1': ['A', 'B'],
        'e2': ['B', 'C'],
        'e3': ['C', 'D'],
        'e4': ['A', 'C']
    }
    
    processor = HypergraphProcessor(initial_edges)
    
    print(f"Initial hypergraph: {processor}")
    print(f"Nodes: {sorted(processor.nodes)}")
    print(f"Edges: {processor.edges}")
    
    return processor


def demonstrate_rule_evolution():
    """Demonstrate hypergraph evolution using rules."""
    print("\n" + "="*50)
    print("RULE-BASED EVOLUTION DEMONSTRATION")
    print("="*50)
    
    # Create initial hypergraph
    processor = create_simple_hypergraph()
    
    # Create rule engine with basic rules
    engine = WolframRuleEngine()
    
    # Add a custom rule for demonstration
    expansion_rule = RewriteRule(
        name="binary_expansion",
        pattern=[("e1", ["A", "B"])],
        replacement=[("e1", ["A", "X"]), ("e2", ["X", "B"]), ("e3", ["A", "X", "B"])],
        priority=2
    )
    
    engine.add_rule(expansion_rule)
    
    # Add basic rules
    basic_rules = WolframRuleEngine.create_basic_rules()
    for rule in basic_rules:
        engine.add_rule(rule)
    
    print(f"\nRule engine initialized with {len(engine.rules)} rules")
    for rule in engine.rules:
        print(f"  - {rule.name} (priority: {rule.priority})")
    
    # Evolve the hypergraph
    print("\nEvolving hypergraph...")
    evolved_processor = engine.evolve(processor, steps=3, max_applications_per_step=2)
    
    print(f"\nAfter evolution: {evolved_processor}")
    print(f"Evolution steps: {evolved_processor.current_step}")
    print(f"Final nodes: {sorted(evolved_processor.nodes)}")
    print(f"Final edges: {evolved_processor.edges}")
    
    # Display evolution statistics
    stats = engine.get_evolution_statistics()
    print(f"\nEvolution statistics:")
    print(f"  Total rule applications: {stats['total_applications']}")
    print(f"  Rule application counts: {stats['rule_application_counts']}")
    
    return evolved_processor, engine


def demonstrate_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*50)
    print("VISUALIZATION DEMONSTRATION")
    print("="*50)
    
    # Create a more complex hypergraph for visualization
    complex_edges = {
        'e1': ['A', 'B'],
        'e2': ['B', 'C'],
        'e3': ['C', 'D'],
        'e4': ['D', 'A'],
        'e5': ['A', 'B', 'C'],  # Hyperedge
        'e6': ['B', 'D', 'E'],  # Another hyperedge
        'e7': ['E', 'F']
    }
    
    processor = HypergraphProcessor(complex_edges)
    visualizer = BasicVisualizer(processor)
    
    print(f"Created complex hypergraph: {processor}")
    
    # Generate static visualization
    print("Generating static visualization...")
    fig = visualizer.plot_static(title="Complex Wolfram Physics Hypergraph")
    plt.savefig("examples/complex_hypergraph.png", dpi=150, bbox_inches='tight')
    print("Static visualization saved as 'examples/complex_hypergraph.png'")
    
    # Demonstrate different layouts
    layouts = ['spring', 'circular', 'random']
    for layout in layouts:
        print(f"Computing {layout} layout...")
        visualizer.compute_layout(layout)
    
    # Show statistics
    stats = processor.compute_statistics()
    print(f"\nHypergraph statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    plt.close('all')  # Close matplotlib figures
    
    return processor, visualizer


def demonstrate_data_management():
    """Demonstrate data storage and retrieval."""
    print("\n" + "="*50)
    print("DATA MANAGEMENT DEMONSTRATION")
    print("="*50)
    
    # Create data manager
    data_manager = DataManager("examples/data_storage", backend='json')
    
    print(f"Data manager initialized: {data_manager}")
    print(f"Storage info: {data_manager.get_storage_info()}")
    
    # Create and save an experiment
    processor = create_simple_hypergraph()
    
    experiment_id = data_manager.save_experiment(
        experiment_name="basic_demo",
        processor=processor,
        description="Basic demonstration of hypergraph evolution",
        parameters={"initial_nodes": 4, "initial_edges": 4}
    )
    
    print(f"\nExperiment saved with ID: {experiment_id}")
    
    # List experiments
    experiments = data_manager.list_experiments()
    print(f"\nStored experiments ({len(experiments)}):")
    for exp in experiments:
        print(f"  - {exp['name']}: {exp['node_count']} nodes, {exp['edge_count']} edges")
    
    # Load and verify experiment
    loaded_processor = data_manager.load_experiment(experiment_id)
    if loaded_processor:
        print(f"\nLoaded experiment: {loaded_processor}")
        print(f"Nodes match: {set(loaded_processor.nodes) == set(processor.nodes)}")
        print(f"Edges match: {loaded_processor.edges == processor.edges}")
    
    return data_manager


def demonstrate_complete_workflow():
    """Demonstrate a complete workflow combining all components."""
    print("\n" + "="*50)
    print("COMPLETE WORKFLOW DEMONSTRATION")
    print("="*50)
    
    # 1. Create initial hypergraph
    processor = create_simple_hypergraph()
    
    # 2. Set up rule engine
    engine = WolframRuleEngine()
    
    # Add custom rule for this demonstration
    merge_rule = RewriteRule(
        name="edge_merger",
        pattern=[("e1", ["A", "B"])],
        replacement=[("e1", ["A", "B", "NEW"])],
        priority=1
    )
    engine.add_rule(merge_rule)
    
    # 3. Create data manager
    data_manager = DataManager("examples/workflow_data", backend='json')
    
    # 4. Create visualizer
    visualizer = BasicVisualizer(processor)
    
    # 5. Save initial state
    initial_id = data_manager.save_experiment(
        "workflow_initial",
        processor,
        "Initial state of workflow demonstration"
    )
    
    # 6. Evolve hypergraph with visualization at each step
    print("\nEvolving hypergraph with step-by-step visualization...")
    
    evolution_history = []
    current_processor = processor
    
    for step in range(3):
        print(f"\nStep {step + 1}:")
        
        # Save current state
        current_processor.save_to_history()
        evolution_history.append(current_processor.snapshot())
        
        # Visualize current state
        visualizer.processor = current_processor
        fig = visualizer.plot_static(title=f"Evolution Step {step}")
        plt.savefig(f"examples/evolution_step_{step}.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Apply evolution
        current_processor = engine.apply_single_step(current_processor)
        
        print(f"  Nodes: {current_processor.node_count}")
        print(f"  Edges: {current_processor.edge_count}")
        print(f"  Complexity: {len(current_processor.edges)}")
    
    # 7. Save final state
    final_id = data_manager.save_experiment(
        "workflow_final",
        current_processor,
        "Final state of workflow demonstration"
    )
    
    # 8. Generate evolution animation
    print("\nGenerating evolution animation...")
    visualizer.processor = current_processor
    anim = visualizer.create_evolution_animation(
        evolution_history,
        interval=1000,
        save_path="examples/evolution_animation.gif"
    )
    
    # 9. Export results
    print("\nExporting results...")
    data_manager.export_experiment(initial_id, "examples/initial_state.json")
    data_manager.export_experiment(final_id, "examples/final_state.json")
    
    # 10. Summary statistics
    final_stats = current_processor.compute_statistics()
    evolution_stats = engine.get_evolution_statistics()
    
    print(f"\nWorkflow completed successfully!")
    print(f"Final hypergraph statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nEvolution statistics:")
    for key, value in evolution_stats.items():
        print(f"  {key}: {value}")
    
    return current_processor, engine, data_manager


def main():
    """Main demonstration function."""
    print("WOLFRAM PHYSICS PROJECT - PHASE 1 DEMONSTRATION")
    print("="*60)
    
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    try:
        # Run individual demonstrations
        evolved_processor, engine = demonstrate_rule_evolution()
        complex_processor, visualizer = demonstrate_visualization()
        data_manager = demonstrate_data_management()
        
        # Run complete workflow
        final_processor, final_engine, final_data_manager = demonstrate_complete_workflow()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print("\nGenerated files:")
        print("  - examples/complex_hypergraph.png")
        print("  - examples/evolution_step_*.png")
        print("  - examples/evolution_animation.gif")
        print("  - examples/initial_state.json")
        print("  - examples/final_state.json")
        print("  - examples/data_storage/ (experiment data)")
        print("  - examples/workflow_data/ (workflow data)")
        
        print("\nPhase 1 implementation is complete and functional!")
        print("Ready for Phase 2: Interactive visualization and controls")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())