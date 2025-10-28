"""Tests for convergence criteria and detection"""

import pytest

from agentcore.dspy_optimization.evolutionary.population import (
    Individual,
    Population,
    PopulationConfig,
)
from agentcore.dspy_optimization.evolutionary.convergence import (
    ConvergenceCriteria,
    ConvergenceDetector,
    ConvergenceReason,
    FitnessPlateauDetector,
    AdaptiveConvergence,
)


class TestFitnessPlateauDetector:
    """Tests for fitness plateau detection"""

    def test_plateau_detection(self) -> None:
        """Test detecting fitness plateau"""
        criteria = ConvergenceCriteria(
            plateau_generations=3,
            plateau_threshold=0.001,
        )
        detector = FitnessPlateauDetector(criteria)

        # Simulate plateau
        for _ in range(5):
            population = Population(PopulationConfig(population_size=10))
            individuals = [
                Individual(genome={}, fitness=0.8)
                for _ in range(10)
            ]
            population.initialize(individuals)
            detector.update(population)

        assert detector.has_converged()
        assert detector.get_plateau_duration() >= 3

    def test_no_plateau_with_improvement(self) -> None:
        """Test no plateau when fitness improves"""
        criteria = ConvergenceCriteria(
            plateau_generations=3,
            plateau_threshold=0.001,
        )
        detector = FitnessPlateauDetector(criteria)

        # Simulate continuous improvement
        for i in range(5):
            population = Population(PopulationConfig(population_size=10))
            individuals = [
                Individual(genome={}, fitness=0.8 + i * 0.1)
                for _ in range(10)
            ]
            population.initialize(individuals)
            detector.update(population)

        assert not detector.has_converged()
        assert detector.get_plateau_duration() == 0

    def test_improvement_rate(self) -> None:
        """Test improvement rate calculation"""
        criteria = ConvergenceCriteria()
        detector = FitnessPlateauDetector(criteria)

        # Add fitness history
        for i in range(10):
            population = Population(PopulationConfig(population_size=10))
            individuals = [
                Individual(genome={}, fitness=0.5 + i * 0.05)
                for _ in range(10)
            ]
            population.initialize(individuals)
            detector.update(population)

        rate = detector.get_improvement_rate(window=5)
        assert rate > 0.0  # Should show improvement


class TestConvergenceDetector:
    """Tests for convergence detector"""

    def test_generation_limit(self) -> None:
        """Test generation limit convergence"""
        criteria = ConvergenceCriteria(max_generations=10)
        detector = ConvergenceDetector(criteria)

        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={}, fitness=0.8) for _ in range(10)]
        population.initialize(individuals)
        population.generation = 10

        status = detector.check_convergence(population)

        assert status.converged
        assert status.reason == ConvergenceReason.GENERATION_LIMIT

    def test_target_fitness(self) -> None:
        """Test target fitness convergence"""
        criteria = ConvergenceCriteria(
            target_fitness=0.9,
            max_generations=100,
        )
        detector = ConvergenceDetector(criteria)

        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={}, fitness=0.95) for _ in range(10)]
        population.initialize(individuals)

        detector.plateau_detector.update(population)
        status = detector.check_convergence(population)

        assert status.converged
        assert status.reason == ConvergenceReason.TARGET_FITNESS

    def test_fitness_plateau_convergence(self) -> None:
        """Test fitness plateau convergence"""
        criteria = ConvergenceCriteria(
            plateau_generations=3,
            max_generations=100,
        )
        detector = ConvergenceDetector(criteria)

        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={}, fitness=0.8) for _ in range(10)]
        population.initialize(individuals)

        # Update multiple times with same fitness
        for _ in range(5):
            detector.plateau_detector.update(population)

        status = detector.check_convergence(population)

        assert status.converged
        assert status.reason == ConvergenceReason.FITNESS_PLATEAU

    def test_diversity_loss(self) -> None:
        """Test diversity loss convergence"""
        criteria = ConvergenceCriteria(
            min_diversity=0.1,
            max_generations=100,
        )
        detector = ConvergenceDetector(criteria)

        # Create population with zero diversity
        population = Population(PopulationConfig(population_size=10))
        individuals = [
            Individual(genome={"param": 0.5}, fitness=0.8)
            for _ in range(10)
        ]
        population.initialize(individuals)

        detector.plateau_detector.update(population)
        status = detector.check_convergence(population)

        assert status.converged
        assert status.reason == ConvergenceReason.DIVERSITY_LOSS

    def test_time_limit(self) -> None:
        """Test time limit convergence"""
        criteria = ConvergenceCriteria(
            max_time_seconds=10,
            max_generations=100,
            min_diversity=0.0,  # Disable diversity check
        )
        start_time = 0.0
        detector = ConvergenceDetector(criteria, start_time=start_time)

        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={"param": idx / 10}, fitness=0.8) for idx in range(10)]
        population.initialize(individuals)

        detector.plateau_detector.update(population)
        current_time = 20.0  # 20 seconds elapsed
        status = detector.check_convergence(population, current_time)

        assert status.converged
        assert status.reason == ConvergenceReason.TIME_LIMIT

    def test_should_continue(self) -> None:
        """Test should_continue method"""
        criteria = ConvergenceCriteria(
            max_generations=10,
            min_diversity=0.0,  # Disable diversity check
        )
        detector = ConvergenceDetector(criteria)

        # Early generation
        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={"param": idx / 10}, fitness=0.8) for idx in range(10)]
        population.initialize(individuals)
        population.generation = 5

        assert detector.should_continue(population)

        # At limit
        population.generation = 10
        assert not detector.should_continue(population)

    def test_progress_info(self) -> None:
        """Test progress information"""
        criteria = ConvergenceCriteria(max_generations=100)
        detector = ConvergenceDetector(criteria)

        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={}, fitness=0.8) for _ in range(10)]
        population.initialize(individuals)
        population.generation = 50

        detector.plateau_detector.update(population)
        progress = detector.get_progress_info(population)

        assert progress["generation"] == 50
        assert progress["max_generations"] == 100
        assert progress["progress_percent"] == 50.0
        assert "best_fitness" in progress
        assert "diversity" in progress


class TestAdaptiveConvergence:
    """Tests for adaptive convergence"""

    def test_adjust_criteria_high_improvement(self) -> None:
        """Test criteria adjustment with high improvement"""
        base_criteria = ConvergenceCriteria(
            plateau_generations=10,
            max_generations=100,
        )
        adaptive = AdaptiveConvergence(base_criteria)

        # Create population with improving fitness
        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={}, fitness=0.8) for _ in range(10)]
        population.initialize(individuals)

        detector = ConvergenceDetector(base_criteria)
        detector.plateau_detector.fitness_history = [0.5, 0.6, 0.7, 0.8]

        adjusted = adaptive.adjust_criteria(population, detector)

        # Should be more lenient with plateau
        assert adjusted.plateau_generations >= base_criteria.plateau_generations

    def test_adjust_criteria_high_diversity(self) -> None:
        """Test criteria adjustment with high diversity"""
        base_criteria = ConvergenceCriteria(
            plateau_generations=10,
            max_generations=100,
        )
        adaptive = AdaptiveConvergence(base_criteria)

        # Create diverse population
        population = Population(PopulationConfig(population_size=10))
        individuals = [
            Individual(genome={"param": i / 10}, fitness=0.8)
            for i in range(10)
        ]
        population.initialize(individuals)

        detector = ConvergenceDetector(base_criteria)
        detector.plateau_detector.update(population)

        adjusted = adaptive.adjust_criteria(population, detector)

        # Should allow more generations with high diversity
        assert adjusted.max_generations >= base_criteria.max_generations

    def test_adjustment_history(self) -> None:
        """Test adjustment history tracking"""
        base_criteria = ConvergenceCriteria()
        adaptive = AdaptiveConvergence(base_criteria)

        population = Population(PopulationConfig(population_size=10))
        individuals = [Individual(genome={}, fitness=0.8) for _ in range(10)]
        population.initialize(individuals)

        detector = ConvergenceDetector(base_criteria)
        detector.plateau_detector.update(population)

        adaptive.adjust_criteria(population, detector)

        history = adaptive.get_adjustment_history()
        assert len(history) == 1
        assert "generation" in history[0]
        assert "plateau_generations" in history[0]
