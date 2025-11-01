"""Tests for selection strategies"""

import pytest

from agentcore.dspy_optimization.evolutionary.population import Individual
from agentcore.dspy_optimization.evolutionary.selection import (
    TournamentSelection,
    RouletteWheelSelection,
    ElitismSelection,
    RankSelection,
    StochasticUniversalSampling,
)


class TestTournamentSelection:
    """Tests for tournament selection"""

    def test_tournament_selection(self) -> None:
        """Test basic tournament selection"""
        population = [
            Individual(genome={}, fitness=float(i))
            for i in range(10)
        ]

        selector = TournamentSelection(tournament_size=3)
        selected = selector.select(population, count=5)

        assert len(selected) == 5
        # Higher fitness individuals should be more likely
        avg_fitness = sum(ind.fitness for ind in selected) / len(selected)
        assert avg_fitness > 4.0  # Should be above average

    def test_tournament_small_population(self) -> None:
        """Test tournament with small population"""
        population = [
            Individual(genome={}, fitness=1.0),
            Individual(genome={}, fitness=2.0),
        ]

        selector = TournamentSelection(tournament_size=5)
        selected = selector.select(population, count=3)

        assert len(selected) == 3


class TestRouletteWheelSelection:
    """Tests for roulette wheel selection"""

    def test_roulette_wheel_selection(self) -> None:
        """Test basic roulette wheel selection"""
        population = [
            Individual(genome={}, fitness=float(i + 1))  # 1-10
            for i in range(10)
        ]

        selector = RouletteWheelSelection()
        selected = selector.select(population, count=5)

        assert len(selected) == 5

    def test_roulette_wheel_zero_fitness(self) -> None:
        """Test roulette wheel with zero fitness"""
        population = [
            Individual(genome={}, fitness=0.0)
            for _ in range(5)
        ]

        selector = RouletteWheelSelection()
        selected = selector.select(population, count=3)

        # Should still select randomly
        assert len(selected) == 3

    def test_roulette_wheel_empty_population(self) -> None:
        """Test roulette wheel with empty population"""
        selector = RouletteWheelSelection()
        selected = selector.select([], count=5)

        assert len(selected) == 0


class TestElitismSelection:
    """Tests for elitism selection"""

    def test_elitism_selection(self) -> None:
        """Test basic elitism selection"""
        population = [
            Individual(genome={}, fitness=float(i))
            for i in range(10)
        ]

        selector = ElitismSelection()
        selected = selector.select(population, count=3)

        assert len(selected) == 3
        assert selected[0].fitness == 9.0  # Highest
        assert selected[1].fitness == 8.0
        assert selected[2].fitness == 7.0

    def test_elitism_more_than_population(self) -> None:
        """Test elitism requesting more than population size"""
        population = [
            Individual(genome={}, fitness=float(i))
            for i in range(5)
        ]

        selector = ElitismSelection()
        selected = selector.select(population, count=10)

        assert len(selected) == 5  # Can't select more than available


class TestRankSelection:
    """Tests for rank selection"""

    def test_rank_selection(self) -> None:
        """Test basic rank selection"""
        population = [
            Individual(genome={}, fitness=float(i))
            for i in range(10)
        ]

        selector = RankSelection(selection_pressure=1.5)
        selected = selector.select(population, count=5)

        assert len(selected) == 5

    def test_rank_selection_empty(self) -> None:
        """Test rank selection with empty population"""
        selector = RankSelection()
        selected = selector.select([], count=5)

        assert len(selected) == 0


class TestStochasticUniversalSampling:
    """Tests for stochastic universal sampling"""

    def test_sus_selection(self) -> None:
        """Test basic SUS selection"""
        population = [
            Individual(genome={}, fitness=float(i + 1))
            for i in range(10)
        ]

        selector = StochasticUniversalSampling()
        selected = selector.select(population, count=5)

        assert len(selected) == 5

    def test_sus_zero_fitness(self) -> None:
        """Test SUS with zero fitness"""
        population = [
            Individual(genome={}, fitness=0.0)
            for _ in range(5)
        ]

        selector = StochasticUniversalSampling()
        selected = selector.select(population, count=3)

        assert len(selected) == 3

    def test_sus_empty(self) -> None:
        """Test SUS with empty population"""
        selector = StochasticUniversalSampling()
        selected = selector.select([], count=5)

        assert len(selected) == 0
