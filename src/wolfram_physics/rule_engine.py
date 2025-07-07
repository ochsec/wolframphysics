"""
Rule engine for Wolfram Physics Project.
Implements rule-based transformation system for hypergraph evolution.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Union, Callable
import random
import copy
from dataclasses import dataclass
from .hypergraph_processor import HypergraphProcessor


@dataclass
class RewriteRule:
    """
    Represents a single rewrite rule for hypergraph transformation.
    """
    name: str
    pattern: List[Tuple[str, List]]  # List of (edge_id, nodes) patterns to match
    replacement: List[Tuple[str, List]]  # List of (edge_id, nodes) to replace with
    conditions: Optional[Callable] = None  # Optional condition function
    priority: int = 1  # Higher priority rules are applied first
    
    def __post_init__(self):
        """Validate rule structure."""
        if not self.pattern:
            raise ValueError("Rule pattern cannot be empty")
        if not self.replacement:
            raise ValueError("Rule replacement cannot be empty")


class WolframRuleEngine:
    """
    Core rule engine for applying Wolfram Physics transformations to hypergraphs.
    Handles pattern matching, rule application, and evolution dynamics.
    """
    
    def __init__(self, rules: Optional[List[RewriteRule]] = None):
        """
        Initialize the rule engine.
        
        Args:
            rules: List of rewrite rules to use
        """
        self.rules = rules or []
        self.application_count = 0
        self.evolution_log = []
        self.random_seed = None
        
    def add_rule(self, rule: RewriteRule) -> None:
        """
        Add a new rewrite rule to the engine.
        
        Args:
            rule: RewriteRule to add
        """
        self.rules.append(rule)
        # Sort rules by priority (higher priority first)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rule by name.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if rule was found and removed, False otherwise
        """
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True
        return False
    
    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible evolution.
        
        Args:
            seed: Random seed value
        """
        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def find_matches(self, processor: HypergraphProcessor, rule: RewriteRule) -> List[Dict]:
        """
        Find all matches for a rule pattern in the hypergraph.
        
        Args:
            processor: HypergraphProcessor containing the hypergraph
            rule: RewriteRule to match
            
        Returns:
            List of match dictionaries containing node mappings
        """
        matches = []
        
        if not rule.pattern:
            return matches
        
        # For simplicity, implement basic pattern matching
        # This is a simplified version - full implementation would need more sophisticated matching
        current_edges = processor.edges
        
        # Try to find subgraph isomorphisms
        for edge_id, edge_nodes in current_edges.items():
            if len(rule.pattern) == 1:  # Single edge pattern
                pattern_edge = rule.pattern[0]
                pattern_edge_name, pattern_nodes = pattern_edge
                
                # Check if sizes match (ignore edge names for pattern matching)
                if len(pattern_nodes) == len(edge_nodes):
                    
                    # For pattern matching, we try to create variable bindings
                    # This allows patterns like ["A", "B"] to match any 2-node edge
                    node_mapping = {}
                    for i, pattern_node in enumerate(pattern_nodes):
                        node_mapping[pattern_node] = edge_nodes[i]
                    
                    # Check conditions if any
                    if rule.conditions is None or rule.conditions(processor, node_mapping):
                        matches.append({
                            'edge_mapping': {pattern_edge_name: edge_id},
                            'node_mapping': node_mapping,
                            'matched_edges': [edge_id]
                        })
        
        return matches
    
    def apply_rule(self, processor: HypergraphProcessor, rule: RewriteRule, 
                   match: Dict) -> HypergraphProcessor:
        """
        Apply a single rule to the hypergraph based on a match.
        
        Args:
            processor: HypergraphProcessor to modify
            rule: RewriteRule to apply
            match: Match dictionary from find_matches
            
        Returns:
            New HypergraphProcessor with rule applied
        """
        new_processor = processor.copy()
        
        # Remove matched edges
        for edge_id in match['matched_edges']:
            new_processor.remove_edge(edge_id)
        
        # Add replacement edges
        node_mapping = match['node_mapping']
        new_node_counter = max([int(n) for n in new_processor.nodes if str(n).isdigit()] + [0]) + 1
        
        for replacement_edge in rule.replacement:
            replacement_id, replacement_nodes = replacement_edge
            
            # Map nodes according to the match
            mapped_nodes = []
            for node in replacement_nodes:
                if node in node_mapping:
                    mapped_nodes.append(node_mapping[node])
                else:
                    # Create new node
                    mapped_nodes.append(str(new_node_counter))
                    new_node_counter += 1
            
            # Generate unique edge ID
            new_edge_id = f"{replacement_id}_{self.application_count}"
            new_processor.add_edge(new_edge_id, mapped_nodes)
        
        new_processor.current_step += 1
        return new_processor
    
    def apply_single_step(self, processor: HypergraphProcessor, 
                         max_applications: int = 1) -> HypergraphProcessor:
        """
        Apply rules for a single evolution step.
        
        Args:
            processor: HypergraphProcessor to evolve
            max_applications: Maximum number of rule applications per step
            
        Returns:
            New HypergraphProcessor after one evolution step
        """
        current_processor = processor.copy()
        applications = 0
        
        # Save state before evolution
        current_processor.save_to_history()
        
        while applications < max_applications:
            rule_applied = False
            
            # Try each rule in priority order
            for rule in self.rules:
                matches = self.find_matches(current_processor, rule)
                
                if matches:
                    # Apply first match (could be randomized)
                    match = matches[0] if len(matches) == 1 else random.choice(matches)
                    current_processor = self.apply_rule(current_processor, rule, match)
                    
                    # Log the application
                    self.evolution_log.append({
                        'step': current_processor.current_step,
                        'rule': rule.name,
                        'match': match,
                        'applications': self.application_count
                    })
                    
                    self.application_count += 1
                    applications += 1
                    rule_applied = True
                    break
            
            # If no rule could be applied, stop
            if not rule_applied:
                break
        
        return current_processor
    
    def evolve(self, processor: HypergraphProcessor, steps: int, 
               max_applications_per_step: int = 1) -> HypergraphProcessor:
        """
        Evolve the hypergraph for multiple steps.
        
        Args:
            processor: HypergraphProcessor to evolve
            steps: Number of evolution steps
            max_applications_per_step: Maximum rule applications per step
            
        Returns:
            HypergraphProcessor after evolution
        """
        current_processor = processor.copy()
        
        for step in range(steps):
            current_processor = self.apply_single_step(
                current_processor, max_applications_per_step
            )
        
        return current_processor
    
    def get_evolution_statistics(self) -> Dict:
        """
        Get statistics about the evolution process.
        
        Returns:
            Dictionary containing evolution statistics
        """
        if not self.evolution_log:
            return {'total_applications': 0}
        
        rule_counts = {}
        for entry in self.evolution_log:
            rule_name = entry['rule']
            rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
        
        return {
            'total_applications': len(self.evolution_log),
            'rule_application_counts': rule_counts,
            'evolution_steps': self.evolution_log[-1]['step'] if self.evolution_log else 0,
            'most_applied_rule': max(rule_counts.items(), key=lambda x: x[1])[0] if rule_counts else None
        }
    
    def reset_statistics(self) -> None:
        """Reset evolution statistics and logs."""
        self.application_count = 0
        self.evolution_log = []
    
    @staticmethod
    def create_basic_rules() -> List[RewriteRule]:
        """
        Create a set of basic Wolfram Physics rules.
        
        Returns:
            List of basic RewriteRule objects
        """
        rules = []
        
        # Rule 1: Binary relation creates triangle
        rules.append(RewriteRule(
            name="binary_to_triangle",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"]), ("e2", ["B", "C"]), ("e3", ["A", "B", "C"])],
            priority=2
        ))
        
        # Rule 2: Triangle simplification
        rules.append(RewriteRule(
            name="triangle_simplification",
            pattern=[("e1", ["A", "B", "C"])],
            replacement=[("e1", ["A", "D"]), ("e2", ["D", "B"])],
            priority=1
        ))
        
        # Rule 3: Node duplication
        rules.append(RewriteRule(
            name="node_duplication",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"]), ("e2", ["C", "B"])],
            priority=1
        ))
        
        return rules
    
    @staticmethod
    def create_advanced_rules() -> List[RewriteRule]:
        """
        Create a set of advanced Wolfram Physics rules.
        
        Returns:
            List of advanced RewriteRule objects
        """
        rules = []
        
        # Rule 1: Causal diamond formation
        rules.append(RewriteRule(
            name="causal_diamond",
            pattern=[("e1", ["A", "B"]), ("e2", ["B", "C"])],
            replacement=[("e1", ["A", "D"]), ("e2", ["D", "E"]), ("e3", ["E", "C"]), ("e4", ["D", "C"])],
            priority=3
        ))
        
        # Rule 2: Hyperedge splitting
        rules.append(RewriteRule(
            name="hyperedge_split",
            pattern=[("e1", ["A", "B", "C", "D"])],
            replacement=[("e1", ["A", "B", "E"]), ("e2", ["E", "C", "D"])],
            priority=2
        ))
        
        # Rule 3: Merge operation
        rules.append(RewriteRule(
            name="edge_merge",
            pattern=[("e1", ["A", "B"]), ("e2", ["B", "C"])],
            replacement=[("e1", ["A", "B", "C"])],
            priority=1
        ))
        
        return rules
    
    def __str__(self) -> str:
        """String representation of the rule engine."""
        return f"WolframRuleEngine(rules={len(self.rules)}, applications={self.application_count})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()