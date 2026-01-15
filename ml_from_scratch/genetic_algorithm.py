"""
Genetic Algorithm for Feature Selection from scratch.

Implements:
- Binary chromosome representation (feature mask)
- Tournament selection
- Single-point and uniform crossover
- Bit-flip mutation
- Elitism

Uses classification accuracy as fitness function.
"""

import numpy as np
from typing import Optional, Callable, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Individual:
    """Represents an individual in the population."""
    chromosome: np.ndarray  # Binary mask of selected features
    fitness: float = 0.0

    @property
    def n_features_selected(self) -> int:
        return int(np.sum(self.chromosome))


class GeneticAlgorithmFeatureSelector:
    """
    Genetic Algorithm for feature subset selection.

    Finds the optimal subset of features that maximizes classifier accuracy.

    Chromosome: Binary array where 1 = feature selected, 0 = feature excluded

    Algorithm:
    1. Initialize random population of feature masks
    2. Evaluate fitness (accuracy) for each individual
    3. Select parents using tournament selection
    4. Apply crossover to create offspring
    5. Apply mutation to offspring
    6. Replace population (with elitism)
    7. Repeat until convergence or max generations

    Attributes:
        best_features_: Boolean mask of best features found
        best_score_: Best fitness score achieved
        n_features_selected_: Number of features in best solution
        history_: Fitness history over generations
    """

    def __init__(self,
                 n_features: int,
                 population_size: int = 50,
                 n_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 tournament_size: int = 3,
                 elite_size: int = 2,
                 min_features: int = 1,
                 max_features: Optional[int] = None,
                 random_state: int = 42):
        """
        Initialize Genetic Algorithm.

        Args:
            n_features: Total number of features
            population_size: Number of individuals in population
            n_generations: Maximum number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
            tournament_size: Number of individuals in tournament selection
            elite_size: Number of best individuals to preserve
            min_features: Minimum number of features to select
            max_features: Maximum number of features (None = all)
            random_state: Random seed
        """
        self.n_features = n_features
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        self.min_features = min_features
        self.max_features = max_features or n_features
        self.random_state = random_state

        self.best_features_: Optional[np.ndarray] = None
        self.best_score_: float = 0.0
        self.n_features_selected_: int = 0
        self.history_: List[Dict] = []
        self._rng = np.random.RandomState(random_state)

    def _initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        population = []

        for _ in range(self.population_size):
            # Random number of features to select
            n_select = self._rng.randint(self.min_features, self.max_features + 1)

            # Random feature selection
            chromosome = np.zeros(self.n_features, dtype=int)
            selected_idx = self._rng.choice(self.n_features, size=n_select, replace=False)
            chromosome[selected_idx] = 1

            population.append(Individual(chromosome=chromosome))

        return population

    def _evaluate_fitness(self, individual: Individual, X: np.ndarray,
                          y: np.ndarray, classifier, cv_folds: int = 3) -> float:
        """
        Evaluate fitness using cross-validation accuracy.

        Args:
            individual: Individual to evaluate
            X: Feature matrix
            y: Labels
            classifier: Classifier class (not instance)
            cv_folds: Number of CV folds

        Returns:
            Fitness score (mean CV accuracy)
        """
        # Get selected features
        mask = individual.chromosome.astype(bool)
        if not np.any(mask):
            return 0.0

        X_selected = X[:, mask]

        # Cross-validation
        n_samples = len(X)
        indices = self._rng.permutation(n_samples)
        fold_size = n_samples // cv_folds

        scores = []
        for fold in range(cv_folds):
            start = fold * fold_size
            end = start + fold_size if fold < cv_folds - 1 else n_samples

            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train and evaluate
            clf = classifier()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy = np.mean(y_pred == y_test)
            scores.append(accuracy)

        # Fitness with slight penalty for more features
        mean_accuracy = np.mean(scores)
        n_selected = individual.n_features_selected
        penalty = 0.001 * n_selected / self.n_features  # Small penalty
        fitness = mean_accuracy - penalty

        return fitness

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = self._rng.choice(population, size=self.tournament_size, replace=False)
        winner = max(tournament, key=lambda ind: ind.fitness)
        return winner

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents.

        Uses single-point crossover.
        """
        if self._rng.random() > self.crossover_rate:
            return (
                Individual(chromosome=parent1.chromosome.copy()),
                Individual(chromosome=parent2.chromosome.copy())
            )

        # Single-point crossover
        point = self._rng.randint(1, self.n_features)

        child1_chrom = np.concatenate([parent1.chromosome[:point], parent2.chromosome[point:]])
        child2_chrom = np.concatenate([parent2.chromosome[:point], parent1.chromosome[point:]])

        return (
            Individual(chromosome=child1_chrom),
            Individual(chromosome=child2_chrom)
        )

    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover - each gene randomly from either parent."""
        if self._rng.random() > self.crossover_rate:
            return (
                Individual(chromosome=parent1.chromosome.copy()),
                Individual(chromosome=parent2.chromosome.copy())
            )

        mask = self._rng.random(self.n_features) < 0.5

        child1_chrom = np.where(mask, parent1.chromosome, parent2.chromosome)
        child2_chrom = np.where(mask, parent2.chromosome, parent1.chromosome)

        return (
            Individual(chromosome=child1_chrom.copy()),
            Individual(chromosome=child2_chrom.copy())
        )

    def _mutate(self, individual: Individual) -> Individual:
        """Apply bit-flip mutation."""
        chromosome = individual.chromosome.copy()

        for i in range(self.n_features):
            if self._rng.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]  # Flip bit

        # Ensure at least min_features are selected
        if np.sum(chromosome) < self.min_features:
            # Add random features
            zeros = np.where(chromosome == 0)[0]
            n_add = self.min_features - int(np.sum(chromosome))
            if len(zeros) >= n_add:
                add_idx = self._rng.choice(zeros, size=n_add, replace=False)
                chromosome[add_idx] = 1

        # Ensure at most max_features are selected
        if np.sum(chromosome) > self.max_features:
            # Remove random features
            ones = np.where(chromosome == 1)[0]
            n_remove = int(np.sum(chromosome)) - self.max_features
            remove_idx = self._rng.choice(ones, size=n_remove, replace=False)
            chromosome[remove_idx] = 0

        return Individual(chromosome=chromosome)

    def fit(self, X: np.ndarray, y: np.ndarray, classifier,
            verbose: bool = True) -> 'GeneticAlgorithmFeatureSelector':
        """
        Run genetic algorithm to find optimal feature subset.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            classifier: Classifier class (e.g., DecisionTreeClassifier)
            verbose: Print progress

        Returns:
            self
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)

        if X.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {X.shape[1]}")

        # Initialize population
        population = self._initialize_population()

        # Evaluate initial population
        for ind in population:
            ind.fitness = self._evaluate_fitness(ind, X, y, classifier)

        # Track best
        self.history_ = []
        best_ever = max(population, key=lambda ind: ind.fitness)

        if verbose:
            print(f"Initial best: {best_ever.fitness:.4f} ({best_ever.n_features_selected} features)")

        # Evolution loop
        for gen in range(self.n_generations):
            # Sort by fitness
            population.sort(key=lambda ind: ind.fitness, reverse=True)

            # Record history
            fitness_values = [ind.fitness for ind in population]
            self.history_.append({
                'generation': gen,
                'best_fitness': fitness_values[0],
                'mean_fitness': np.mean(fitness_values),
                'best_n_features': population[0].n_features_selected
            })

            # Elitism - keep best individuals
            new_population = population[:self.elite_size]

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                # Evaluate
                child1.fitness = self._evaluate_fitness(child1, X, y, classifier)
                child2.fitness = self._evaluate_fitness(child2, X, y, classifier)

                new_population.extend([child1, child2])

            # Trim to population size
            population = new_population[:self.population_size]

            # Update best
            current_best = max(population, key=lambda ind: ind.fitness)
            if current_best.fitness > best_ever.fitness:
                best_ever = Individual(
                    chromosome=current_best.chromosome.copy(),
                    fitness=current_best.fitness
                )

            if verbose and (gen + 1) % 10 == 0:
                print(f"Gen {gen + 1}: best={current_best.fitness:.4f}, "
                      f"mean={np.mean([ind.fitness for ind in population]):.4f}, "
                      f"features={current_best.n_features_selected}")

        # Store best solution
        self.best_features_ = best_ever.chromosome.astype(bool)
        self.best_score_ = best_ever.fitness
        self.n_features_selected_ = best_ever.n_features_selected

        if verbose:
            print(f"\nBest solution: {self.best_score_:.4f} accuracy "
                  f"with {self.n_features_selected_} features")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select features based on best solution.

        Args:
            X: Feature matrix

        Returns:
            Selected features
        """
        if self.best_features_ is None:
            raise ValueError("GA not fitted. Call fit() first.")

        return np.array(X)[:, self.best_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, classifier,
                      verbose: bool = True) -> np.ndarray:
        """Fit GA and return transformed features."""
        self.fit(X, y, classifier, verbose)
        return self.transform(X)

    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.best_features_ is None:
            raise ValueError("GA not fitted. Call fit() first.")
        return np.where(self.best_features_)[0]

    def get_feature_importance(self, n_runs: int = 5, X: np.ndarray = None,
                               y: np.ndarray = None, classifier = None) -> np.ndarray:
        """
        Estimate feature importance by running GA multiple times.

        Args:
            n_runs: Number of GA runs
            X, y, classifier: Data and classifier for additional runs

        Returns:
            Feature importance (selection frequency)
        """
        if self.best_features_ is None:
            raise ValueError("GA not fitted. Call fit() first.")

        importance = self.best_features_.astype(float)

        if X is not None and y is not None and classifier is not None:
            for run in range(n_runs - 1):
                ga = GeneticAlgorithmFeatureSelector(
                    n_features=self.n_features,
                    population_size=self.population_size,
                    n_generations=self.n_generations // 2,  # Faster
                    random_state=self.random_state + run + 1
                )
                ga.fit(X, y, classifier, verbose=False)
                importance += ga.best_features_.astype(float)

            importance /= n_runs

        return importance


class SimpleGA:
    """
    Simplified GA wrapper for quick feature selection.
    """

    def __init__(self, n_generations: int = 50, population_size: int = 30):
        self.n_generations = n_generations
        self.population_size = population_size
        self.best_features_: Optional[np.ndarray] = None
        self.best_score_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, classifier) -> 'SimpleGA':
        """Run GA feature selection."""
        n_features = X.shape[1]
        ga = GeneticAlgorithmFeatureSelector(
            n_features=n_features,
            population_size=self.population_size,
            n_generations=self.n_generations
        )
        ga.fit(X, y, classifier, verbose=False)
        self.best_features_ = ga.best_features_
        self.best_score_ = ga.best_score_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using selected features."""
        return np.array(X)[:, self.best_features_]


# =============================================================================
# Test functions
# =============================================================================

def test_genetic_algorithm():
    """Test GA feature selection."""
    print("Testing Genetic Algorithm Feature Selection...")

    np.random.seed(42)

    # Generate data with some irrelevant features
    n_samples = 300
    n_relevant = 5
    n_irrelevant = 10
    n_features = n_relevant + n_irrelevant

    # Relevant features (correlated with class)
    X_relevant = np.random.randn(n_samples, n_relevant)
    y = (X_relevant[:, 0] + X_relevant[:, 1] > 0).astype(int)

    # Irrelevant features (noise)
    X_irrelevant = np.random.randn(n_samples, n_irrelevant)

    # Combine
    X = np.hstack([X_relevant, X_irrelevant])

    # Shuffle features
    perm = np.random.permutation(n_features)
    X = X[:, perm]
    relevant_mask = np.zeros(n_features, dtype=bool)
    relevant_mask[perm < n_relevant] = True

    print(f"Data: {n_samples} samples, {n_features} features")
    print(f"Relevant features at indices: {np.where(relevant_mask)[0]}")

    # Run GA
    from ml_from_scratch.decision_tree import DecisionTreeClassifier

    ga = GeneticAlgorithmFeatureSelector(
        n_features=n_features,
        population_size=30,
        n_generations=30,
        random_state=42
    )

    ga.fit(X, y, DecisionTreeClassifier, verbose=True)

    print(f"\nSelected features: {ga.get_selected_features()}")
    print(f"Overlap with relevant: {np.sum(ga.best_features_ & relevant_mask)}/{n_relevant}")


if __name__ == "__main__":
    test_genetic_algorithm()
