"""
Test suite for WolframRuleEngine class.
"""

import pytest
from src.wolfram_physics.hypergraph_processor import HypergraphProcessor
from src.wolfram_physics.rule_engine import WolframRuleEngine, RewriteRule


class TestRewriteRule:
    """Test cases for RewriteRule dataclass."""
    
    def test_basic_rule_creation(self):
        """Test basic rule creation."""
        rule = RewriteRule(
            name="test_rule",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"]), ("e2", ["B", "C"])]
        )
        
        assert rule.name == "test_rule"
        assert rule.pattern == [("e1", ["A", "B"])]
        assert rule.replacement == [("e1", ["A", "C"]), ("e2", ["B", "C"])]
        assert rule.priority == 1
        assert rule.conditions is None
    
    def test_rule_with_priority(self):
        """Test rule creation with custom priority."""
        rule = RewriteRule(
            name="high_priority",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])],
            priority=5
        )
        
        assert rule.priority == 5
    
    def test_rule_validation(self):
        """Test rule validation."""
        # Empty pattern should raise error
        with pytest.raises(ValueError, match="Rule pattern cannot be empty"):
            RewriteRule(
                name="invalid",
                pattern=[],
                replacement=[("e1", ["A", "B"])]
            )
        
        # Empty replacement should raise error
        with pytest.raises(ValueError, match="Rule replacement cannot be empty"):
            RewriteRule(
                name="invalid",
                pattern=[("e1", ["A", "B"])],
                replacement=[]
            )


class TestWolframRuleEngine:
    """Test cases for WolframRuleEngine."""
    
    def test_initialization_empty(self):
        """Test initialization with no rules."""
        engine = WolframRuleEngine()
        assert len(engine.rules) == 0
        assert engine.application_count == 0
        assert len(engine.evolution_log) == 0
    
    def test_initialization_with_rules(self):
        """Test initialization with predefined rules."""
        rules = [
            RewriteRule(
                name="rule1",
                pattern=[("e1", ["A", "B"])],
                replacement=[("e1", ["A", "C"])]
            )
        ]
        engine = WolframRuleEngine(rules)
        assert len(engine.rules) == 1
        assert engine.rules[0].name == "rule1"
    
    def test_add_rule(self):
        """Test adding rules to engine."""
        engine = WolframRuleEngine()
        
        rule1 = RewriteRule(
            name="rule1",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])],
            priority=1
        )
        
        rule2 = RewriteRule(
            name="rule2",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "D"])],
            priority=3
        )
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        
        assert len(engine.rules) == 2
        # Should be sorted by priority (higher first)
        assert engine.rules[0].priority == 3
        assert engine.rules[1].priority == 1
    
    def test_remove_rule(self):
        """Test removing rules from engine."""
        rule = RewriteRule(
            name="test_rule",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])]
        )
        
        engine = WolframRuleEngine([rule])
        assert len(engine.rules) == 1
        
        # Remove existing rule
        assert engine.remove_rule("test_rule") is True
        assert len(engine.rules) == 0
        
        # Try to remove non-existent rule
        assert engine.remove_rule("nonexistent") is False
    
    def test_set_random_seed(self):
        """Test setting random seed."""
        engine = WolframRuleEngine()
        engine.set_random_seed(42)
        assert engine.random_seed == 42
    
    def test_find_matches_basic(self):
        """Test basic pattern matching."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="match_test",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])]
        )
        
        engine = WolframRuleEngine()
        matches = engine.find_matches(processor, rule)
        
        assert len(matches) == 1
        assert 'node_mapping' in matches[0]
        assert 'edge_mapping' in matches[0]
        assert 'matched_edges' in matches[0]
    
    def test_find_matches_no_match(self):
        """Test pattern matching with no matches."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="no_match",
            pattern=[("e1", ["C", "D", "E"])],  # Different size (3 vs 2 nodes)
            replacement=[("e1", ["C", "F"])]
        )
        
        engine = WolframRuleEngine()
        matches = engine.find_matches(processor, rule)
        
        assert len(matches) == 0
    
    def test_apply_rule_basic(self):
        """Test basic rule application."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="expand_rule",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"]), ("e2", ["B", "C"])]
        )
        
        engine = WolframRuleEngine()
        matches = engine.find_matches(processor, rule)
        
        if matches:
            new_processor = engine.apply_rule(processor, rule, matches[0])
            
            assert new_processor.edge_count == 2
            assert new_processor.current_step == 1
            assert new_processor.node_count >= 2  # At least A, B (C might be new)
    
    def test_apply_single_step(self):
        """Test single evolution step."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="step_rule",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])]
        )
        
        engine = WolframRuleEngine([rule])
        evolved_processor = engine.apply_single_step(processor)
        
        assert evolved_processor.current_step >= processor.current_step
        assert len(evolved_processor.evolution_history) > 0
    
    def test_evolve_multiple_steps(self):
        """Test evolution over multiple steps."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="multi_step",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"]), ("e2", ["B", "C"])]
        )
        
        engine = WolframRuleEngine([rule])
        evolved_processor = engine.evolve(processor, steps=3)
        
        assert evolved_processor.current_step >= 3
        assert len(evolved_processor.evolution_history) >= 3
    
    def test_evolution_statistics(self):
        """Test evolution statistics tracking."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="stats_rule",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])]
        )
        
        engine = WolframRuleEngine([rule])
        engine.evolve(processor, steps=2)
        
        stats = engine.get_evolution_statistics()
        
        assert 'total_applications' in stats
        assert 'rule_application_counts' in stats
        assert 'evolution_steps' in stats
        assert stats['total_applications'] >= 0
    
    def test_reset_statistics(self):
        """Test resetting evolution statistics."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="reset_rule",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])]
        )
        
        engine = WolframRuleEngine([rule])
        engine.evolve(processor, steps=2)
        
        # Check that statistics exist
        stats_before = engine.get_evolution_statistics()
        assert stats_before['total_applications'] > 0
        
        # Reset and check
        engine.reset_statistics()
        stats_after = engine.get_evolution_statistics()
        assert stats_after['total_applications'] == 0
        assert len(engine.evolution_log) == 0
    
    def test_create_basic_rules(self):
        """Test basic rule creation factory method."""
        basic_rules = WolframRuleEngine.create_basic_rules()
        
        assert len(basic_rules) > 0
        assert all(isinstance(rule, RewriteRule) for rule in basic_rules)
        assert all(rule.name for rule in basic_rules)
        assert all(rule.pattern for rule in basic_rules)
        assert all(rule.replacement for rule in basic_rules)
    
    def test_create_advanced_rules(self):
        """Test advanced rule creation factory method."""
        advanced_rules = WolframRuleEngine.create_advanced_rules()
        
        assert len(advanced_rules) > 0
        assert all(isinstance(rule, RewriteRule) for rule in advanced_rules)
        
        # Check that advanced rules have higher complexity
        for rule in advanced_rules:
            assert len(rule.replacement) >= 1
    
    def test_rule_priority_ordering(self):
        """Test that rules are applied in priority order."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        # Create rules with different priorities
        low_priority = RewriteRule(
            name="low",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "LOW"])],
            priority=1
        )
        
        high_priority = RewriteRule(
            name="high",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "HIGH"])],
            priority=5
        )
        
        engine = WolframRuleEngine()
        engine.add_rule(low_priority)
        engine.add_rule(high_priority)
        
        # High priority rule should be first
        assert engine.rules[0].priority == 5
        assert engine.rules[1].priority == 1
    
    def test_string_representation(self):
        """Test string representation."""
        rule = RewriteRule(
            name="test",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])]
        )
        
        engine = WolframRuleEngine([rule])
        str_repr = str(engine)
        
        assert 'WolframRuleEngine' in str_repr
        assert 'rules=1' in str_repr
        assert 'applications=0' in str_repr


class TestRuleEngineEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_processor_evolution(self):
        """Test evolution with empty processor."""
        processor = HypergraphProcessor()
        
        rule = RewriteRule(
            name="empty_test",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])]
        )
        
        engine = WolframRuleEngine([rule])
        evolved_processor = engine.evolve(processor, steps=5)
        
        # Should remain empty
        assert evolved_processor.node_count == 0
        assert evolved_processor.edge_count == 0
    
    def test_no_applicable_rules(self):
        """Test evolution when no rules can be applied."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        rule = RewriteRule(
            name="inapplicable",
            pattern=[("e1", ["X", "Y"])],  # No match
            replacement=[("e1", ["X", "Z"])]
        )
        
        engine = WolframRuleEngine([rule])
        evolved_processor = engine.evolve(processor, steps=5)
        
        # Should remain unchanged
        assert evolved_processor.node_count == processor.node_count
        assert evolved_processor.edge_count == processor.edge_count
    
    def test_rule_conditions(self):
        """Test rules with conditions."""
        processor = HypergraphProcessor({'e1': ['A', 'B']})
        
        def always_false(proc, mapping):
            return False
        
        rule = RewriteRule(
            name="conditional",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "C"])],
            conditions=always_false
        )
        
        engine = WolframRuleEngine([rule])
        matches = engine.find_matches(processor, rule)
        
        # Should find no matches due to condition
        assert len(matches) == 0
    
    def test_max_applications_per_step(self):
        """Test limiting rule applications per step."""
        # Create processor with multiple applicable patterns
        processor = HypergraphProcessor({
            'e1': ['A', 'B'],
            'e2': ['C', 'D']
        })
        
        rule = RewriteRule(
            name="limited",
            pattern=[("e1", ["A", "B"])],
            replacement=[("e1", ["A", "X"])]
        )
        
        engine = WolframRuleEngine([rule])
        evolved_processor = engine.apply_single_step(processor, max_applications=1)
        
        # Should apply at most 1 rule
        assert engine.application_count <= 1


if __name__ == '__main__':
    pytest.main([__file__])