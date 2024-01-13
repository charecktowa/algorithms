import random


# One Max Problem fitness function
def fitness(individual):
    return sum(individual)


# Generate a random individual
def generate_individual(length):
    return [random.randint(0, 1) for _ in range(length)]


# Generate a population of random individuals
def generate_population(population_size, individual_length):
    return [generate_individual(individual_length) for _ in range(population_size)]


# Select parents for crossover using tournament selection
def tournament_selection(population, tournament_size):
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=fitness)


# Perform single-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Perform mutation by flipping a random bit
def mutate(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
    return mutated_individual


# Genetic algorithm for the One Max Problem
def genetic_algorithm(
    population_size, individual_length, tournament_size, mutation_rate, generations
):
    population = generate_population(population_size, individual_length)

    for _ in range(generations):
        new_population = []

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, tournament_size)
            parent2 = tournament_selection(population, tournament_size)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)

            new_population.append(child1)
            new_population.append(child2)

        population = new_population

    best_individual = max(population, key=fitness)
    return best_individual


# Example usage
population_size = 100
individual_length = 10
tournament_size = 5
mutation_rate = 0.01
generations = 100

best_individual = genetic_algorithm(
    population_size, individual_length, tournament_size, mutation_rate, generations
)
print("Best individual:", best_individual)
print("Fitness:", fitness(best_individual))
