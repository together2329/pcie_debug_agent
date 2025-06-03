#!/usr/bin/env python3
"""
Portfolio Optimizer Implementations
Specialized optimizer classes for Bayesian Optimization, CMA-ES, and Population-Based Training
"""

import numpy as np
import random
from typing import Dict, List, Any, Callable, Tuple, Optional
from dataclasses import dataclass, field
import math
from scipy.stats import norm
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Result of an optimization step"""
    params: Dict[str, Any]
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class BayesianOptimizer:
    """
    Gaussian Process-based Bayesian Optimization
    Uses Expected Improvement (EI) acquisition function
    """
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 exploration_weight: float = 0.1):
        self.param_bounds = param_bounds
        self.exploration_weight = exploration_weight
        
        # Gaussian Process components
        self.observations = []
        self.kernel_bandwidth = 1.0
        self.noise_variance = 0.01
        
        # Acquisition function parameters
        self.xi = 0.01  # Exploration parameter for EI
        self.kappa = 2.576  # Exploration parameter for UCB
        
    def suggest_next_points(self, n_suggestions: int, 
                           temperature: float = 1.0) -> List[Dict[str, Any]]:
        """Suggest next points to evaluate based on acquisition function"""
        suggestions = []
        
        if len(self.observations) < 2:
            # Random exploration when we have few observations
            for _ in range(n_suggestions):
                suggestion = self._random_sample()
                suggestions.append(suggestion)
        else:
            # Use acquisition function
            for _ in range(n_suggestions):
                # Find point with highest expected improvement
                best_ei = -float('inf')
                best_point = None
                
                # Random search for acquisition function maximum
                for _ in range(100):
                    candidate = self._random_sample()
                    ei = self._expected_improvement(candidate, temperature)
                    
                    if ei > best_ei:
                        best_ei = ei
                        best_point = candidate
                
                suggestions.append(best_point)
        
        return suggestions
    
    def update(self, params: Dict[str, Any], score: float):
        """Update GP with new observation"""
        self.observations.append({'params': params, 'score': score})
        
        # Adapt kernel bandwidth based on data
        if len(self.observations) > 10:
            self._adapt_kernel_bandwidth()
    
    def _random_sample(self) -> Dict[str, Any]:
        """Generate random sample within bounds"""
        sample = {}
        for param, (low, high) in self.param_bounds.items():
            if isinstance(low, int) and isinstance(high, int):
                sample[param] = random.randint(low, high)
            else:
                sample[param] = random.uniform(low, high)
        return sample
    
    def _gaussian_kernel(self, x1: Dict[str, Any], x2: Dict[str, Any]) -> float:
        """RBF kernel for Gaussian Process"""
        # Convert to vectors
        v1 = np.array([x1.get(k, 0) for k in sorted(self.param_bounds.keys())])
        v2 = np.array([x2.get(k, 0) for k in sorted(self.param_bounds.keys())])
        
        # RBF kernel: k(x1, x2) = exp(-||x1 - x2||² / (2 * bandwidth²))
        distance_sq = np.sum((v1 - v2) ** 2)
        return np.exp(-distance_sq / (2 * self.kernel_bandwidth ** 2))
    
    def _predict(self, x: Dict[str, Any]) -> Tuple[float, float]:
        """Predict mean and variance at point x using GP"""
        if not self.observations:
            return 0.0, 1.0
        
        # Compute kernel vector
        k_star = np.array([self._gaussian_kernel(x, obs['params']) 
                          for obs in self.observations])
        
        # Compute kernel matrix
        n = len(self.observations)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._gaussian_kernel(
                    self.observations[i]['params'],
                    self.observations[j]['params']
                )
        
        # Add noise to diagonal
        K += self.noise_variance * np.eye(n)
        
        # Compute mean and variance
        y = np.array([obs['score'] for obs in self.observations])
        
        try:
            K_inv = np.linalg.inv(K)
            mean = k_star.T @ K_inv @ y
            variance = 1.0 - k_star.T @ K_inv @ k_star
            variance = max(0.0001, variance)  # Ensure positive
        except np.linalg.LinAlgError:
            # Fallback for singular matrix
            mean = np.mean(y)
            variance = np.var(y) + 0.0001
        
        return float(mean), float(variance)
    
    def _expected_improvement(self, x: Dict[str, Any], temperature: float) -> float:
        """Calculate Expected Improvement acquisition function"""
        mean, variance = self._predict(x)
        std = np.sqrt(variance)
        
        # Current best
        if self.observations:
            f_best = max(obs['score'] for obs in self.observations)
        else:
            f_best = 0.0
        
        # Expected Improvement
        with np.errstate(divide='warn'):
            improvement = mean - f_best - self.xi * temperature
            Z = improvement / std if std > 0 else 0.0
            ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        
        return ei
    
    def _adapt_kernel_bandwidth(self):
        """Adapt kernel bandwidth based on data density"""
        # Estimate characteristic length scale
        if len(self.observations) > 1:
            distances = []
            for i in range(len(self.observations) - 1):
                for j in range(i + 1, len(self.observations)):
                    v1 = np.array([self.observations[i]['params'].get(k, 0) 
                                  for k in sorted(self.param_bounds.keys())])
                    v2 = np.array([self.observations[j]['params'].get(k, 0) 
                                  for k in sorted(self.param_bounds.keys())])
                    distances.append(np.linalg.norm(v1 - v2))
            
            if distances:
                # Use median distance as bandwidth
                self.kernel_bandwidth = np.median(distances)


class CMAESOptimizer:
    """
    Covariance Matrix Adaptation Evolution Strategy
    State-of-the-art for continuous optimization
    """
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 population_size: Optional[int] = None):
        self.param_bounds = param_bounds
        self.dimension = len(param_bounds)
        
        # Population size (default: 4 + floor(3 * log(n)))
        if population_size is None:
            self.population_size = 4 + int(3 * np.log(self.dimension))
        else:
            self.population_size = population_size
        
        # Parent number
        self.mu = self.population_size // 2
        
        # Initialize strategy parameters
        self.mean = np.zeros(self.dimension)
        self.sigma = 0.3  # Step size
        self.covariance = np.eye(self.dimension)
        
        # Evolution paths
        self.pc = np.zeros(self.dimension)  # Covariance path
        self.ps = np.zeros(self.dimension)  # Step size path
        
        # Initialize weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        
        # Learning rates
        self.cc = 4.0 / (self.dimension + 4.0)
        self.cs = (self.mueff + 2.0) / (self.dimension + self.mueff + 3.0)
        self.c1 = 2.0 / ((self.dimension + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 
                      2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / 
                      ((self.dimension + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) / 
                                                  (self.dimension + 1.0)) - 1.0) + self.cs
        
        # Generation counter
        self.generation = 0
        
    def suggest_next_points(self, n_suggestions: int, 
                           temperature: float = 1.0) -> List[Dict[str, Any]]:
        """Generate new population"""
        suggestions = []
        
        # Eigendecomposition for efficient sampling
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure positive
        
        for _ in range(n_suggestions):
            # Sample from N(0, C)
            z = np.random.randn(self.dimension)
            y = eigenvectors @ (np.sqrt(eigenvalues) * z)
            x = self.mean + self.sigma * temperature * y
            
            # Clip to bounds and convert to dict
            x_dict = {}
            param_names = sorted(self.param_bounds.keys())
            for i, param in enumerate(param_names):
                low, high = self.param_bounds[param]
                value = np.clip(x[i], low, high)
                if isinstance(low, int) and isinstance(high, int):
                    x_dict[param] = int(np.round(value))
                else:
                    x_dict[param] = float(value)
            
            suggestions.append(x_dict)
        
        return suggestions
    
    def update(self, population: List[Tuple[Dict[str, Any], float]]):
        """Update CMA-ES with evaluated population"""
        # Sort by fitness (descending)
        population.sort(key=lambda x: x[1], reverse=True)
        
        # Extract parameters
        param_names = sorted(self.param_bounds.keys())
        x_array = np.array([[ind[0][param] for param in param_names] 
                           for ind in population])
        
        # Select parents
        parents = x_array[:self.mu]
        
        # Update mean
        old_mean = self.mean.copy()
        self.mean = self.weights @ parents
        
        # Update evolution paths
        y_mean = (self.mean - old_mean) / self.sigma
        
        # Cumulation for sigma
        self.ps = (1 - self.cs) * self.ps + \
                 np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                 np.linalg.inv(np.linalg.cholesky(self.covariance)) @ y_mean
        
        # Cumulation for covariance
        hsig = (np.linalg.norm(self.ps) / 
                np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / 
                np.sqrt(self.dimension) < 
                1.4 + 2 / (self.dimension + 1))
        
        self.pc = (1 - self.cc) * self.pc + \
                 hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean
        
        # Update covariance
        # Rank-one update
        c1_update = self.c1 * (np.outer(self.pc, self.pc) - 
                              (1 - hsig) * self.cc * (2 - self.cc) * self.covariance)
        
        # Rank-mu update
        y_parents = (parents - old_mean) / self.sigma
        cmu_update = self.cmu * sum(self.weights[i] * np.outer(y_parents[i], y_parents[i])
                                   for i in range(self.mu))
        
        self.covariance = (1 - self.c1 - self.cmu) * self.covariance + c1_update + cmu_update
        
        # Update step size
        self.sigma *= np.exp((self.cs / self.damps) * 
                            (np.linalg.norm(self.ps) / np.sqrt(self.dimension) - 1))
        
        self.generation += 1


class PopulationBasedTraining:
    """
    Population-Based Training with evolutionary operators
    Combines evolution strategies with hyperparameter scheduling
    """
    
    def __init__(self, param_bounds: Dict[str, Tuple[float, float]], 
                 population_size: int = 20):
        self.param_bounds = param_bounds
        self.population_size = population_size
        
        # Population management
        self.population = []
        self.elite_size = max(2, population_size // 5)
        
        # Evolutionary parameters
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.tournament_size = 3
        
        # Diversity maintenance
        self.diversity_threshold = 0.1
        self.speciation_threshold = 0.3
        
        # Performance tracking
        self.generation = 0
        self.stagnation_counter = 0
        
    def initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize diverse population"""
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, (low, high) in self.param_bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    individual[param] = random.randint(low, high)
                else:
                    individual[param] = random.uniform(low, high)
            population.append(individual)
        
        self.population = population
        return population
    
    def suggest_next_points(self, n_suggestions: int, 
                           temperature: float = 1.0) -> List[Dict[str, Any]]:
        """Generate offspring through evolutionary operators"""
        if not self.population:
            return self.initialize_population()[:n_suggestions]
        
        offspring = []
        
        while len(offspring) < n_suggestions:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            child1 = self._mutate(child1, temperature)
            child2 = self._mutate(child2, temperature)
            
            offspring.extend([child1, child2])
        
        # Ensure diversity
        offspring = self._ensure_diversity(offspring)
        
        return offspring[:n_suggestions]
    
    def update(self, evaluated_population: List[Tuple[Dict[str, Any], float]]):
        """Update population with elitism and diversity preservation"""
        # Combine with existing population
        all_individuals = []
        
        # Add evaluated individuals
        for params, score in evaluated_population:
            all_individuals.append({'params': params, 'score': score})
        
        # Add existing population with their scores (if available)
        for individual in self.population:
            # Assume we have scores stored or use a default
            all_individuals.append({
                'params': individual, 
                'score': getattr(individual, '_score', 0.0)
            })
        
        # Sort by fitness
        all_individuals.sort(key=lambda x: x['score'], reverse=True)
        
        # Select new population with elitism
        new_population = []
        
        # Keep elite
        for i in range(min(self.elite_size, len(all_individuals))):
            params = all_individuals[i]['params'].copy()
            params._score = all_individuals[i]['score']  # Store score
            new_population.append(params)
        
        # Fill rest with diversity consideration
        candidates = all_individuals[self.elite_size:]
        while len(new_population) < self.population_size and candidates:
            # Select diverse individuals
            selected = self._select_diverse_individual(new_population, candidates)
            if selected:
                params = selected['params'].copy()
                params._score = selected['score']
                new_population.append(params)
                candidates.remove(selected)
            else:
                break
        
        # Update population
        self.population = new_population
        self.generation += 1
        
        # Check stagnation
        if len(all_individuals) > 1:
            best_score = all_individuals[0]['score']
            if hasattr(self, 'prev_best_score') and abs(best_score - self.prev_best_score) < 0.001:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
            self.prev_best_score = best_score
        
        # Inject diversity if stagnating
        if self.stagnation_counter > 10:
            self._inject_diversity()
            self.stagnation_counter = 0
    
    def _tournament_selection(self) -> Dict[str, Any]:
        """Select individual through tournament"""
        tournament = random.sample(self.population, 
                                 min(self.tournament_size, len(self.population)))
        
        # Select based on fitness (stored as _score attribute)
        best = max(tournament, key=lambda x: getattr(x, '_score', 0.0))
        return best.copy()
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Perform crossover between parents"""
        child1, child2 = {}, {}
        
        for param in self.param_bounds:
            if random.random() < 0.5:
                # Uniform crossover
                child1[param] = parent1.get(param, 0)
                child2[param] = parent2.get(param, 0)
            else:
                # Arithmetic crossover
                alpha = random.random()
                if isinstance(parent1.get(param, 0), int):
                    child1[param] = int(alpha * parent1.get(param, 0) + 
                                       (1 - alpha) * parent2.get(param, 0))
                    child2[param] = int((1 - alpha) * parent1.get(param, 0) + 
                                       alpha * parent2.get(param, 0))
                else:
                    child1[param] = alpha * parent1.get(param, 0) + \
                                   (1 - alpha) * parent2.get(param, 0)
                    child2[param] = (1 - alpha) * parent1.get(param, 0) + \
                                   alpha * parent2.get(param, 0)
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, Any], temperature: float) -> Dict[str, Any]:
        """Apply mutation to individual"""
        mutated = individual.copy()
        
        for param, (low, high) in self.param_bounds.items():
            if random.random() < self.mutation_rate * temperature:
                if isinstance(low, int) and isinstance(high, int):
                    # Integer mutation
                    range_size = high - low
                    mutation = int(random.gauss(0, range_size * 0.1))
                    mutated[param] = np.clip(individual.get(param, low) + mutation, low, high)
                else:
                    # Continuous mutation
                    range_size = high - low
                    mutation = random.gauss(0, range_size * 0.1 * temperature)
                    mutated[param] = np.clip(individual.get(param, low) + mutation, low, high)
        
        return mutated
    
    def _ensure_diversity(self, population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure population diversity"""
        diverse_population = []
        
        for individual in population:
            if not diverse_population or self._is_diverse(individual, diverse_population):
                diverse_population.append(individual)
        
        return diverse_population
    
    def _is_diverse(self, individual: Dict[str, Any], 
                    population: List[Dict[str, Any]]) -> bool:
        """Check if individual is diverse from population"""
        for other in population:
            distance = self._parameter_distance(individual, other)
            if distance < self.diversity_threshold:
                return False
        return True
    
    def _parameter_distance(self, ind1: Dict[str, Any], ind2: Dict[str, Any]) -> float:
        """Calculate normalized distance between individuals"""
        distance = 0.0
        n_params = 0
        
        for param, (low, high) in self.param_bounds.items():
            if param in ind1 and param in ind2:
                range_size = high - low
                if range_size > 0:
                    diff = abs(ind1[param] - ind2[param]) / range_size
                    distance += diff ** 2
                    n_params += 1
        
        return np.sqrt(distance / max(n_params, 1))
    
    def _select_diverse_individual(self, current_pop: List[Dict[str, Any]], 
                                  candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select individual that maximizes diversity"""
        best_diversity = 0.0
        best_candidate = None
        
        for candidate in candidates:
            min_distance = float('inf')
            
            for individual in current_pop:
                distance = self._parameter_distance(
                    candidate['params'], 
                    individual
                )
                min_distance = min(min_distance, distance)
            
            if min_distance > best_diversity:
                best_diversity = min_distance
                best_candidate = candidate
        
        return best_candidate if best_diversity > self.diversity_threshold else None
    
    def _inject_diversity(self):
        """Inject random individuals to increase diversity"""
        n_inject = max(2, self.population_size // 10)
        
        for i in range(n_inject):
            if i < len(self.population):
                # Replace worst individuals
                new_individual = {}
                for param, (low, high) in self.param_bounds.items():
                    if isinstance(low, int) and isinstance(high, int):
                        new_individual[param] = random.randint(low, high)
                    else:
                        new_individual[param] = random.uniform(low, high)
                
                new_individual._score = 0.0  # Unknown score
                self.population[-(i+1)] = new_individual