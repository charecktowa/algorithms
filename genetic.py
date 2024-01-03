import numpy as np


# Define the fitness function
def fitness_function(population):
    return np.sum(population, axis=1)


# Define the selection operator (tournament selection)
def selection(population, fitness_values, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        # Select two random individuals
        random_indices = np.random.randint(0, population.shape[0], size=2)
        tournament = fitness_values[random_indices]
        # Select the individual with the highest fitness
        winner_index = np.argmax(tournament)
        parents[i] = population[random_indices[winner_index]]
    return parents


# Define the crossover operator (single-point crossover)
def crossover(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    for i in range(num_offsprings):
        # Select two random parents
        random_indices = np.random.randint(0, parents.shape[0], size=2)
        parent1, parent2 = parents[random_indices]
        # Select a random crossover point
        crossover_point = np.random.randint(0, parents.shape[1])
        # Create the offspring by combining the parents' genes
        offspring = np.concatenate(
            (parent1[:crossover_point], parent2[crossover_point:])
        )
        offsprings[i] = offspring
    return offsprings


# Define the mutation operator (bit-flip mutation)
def mutation(offsprings, mutation_rate):
    for i in range(offsprings.shape[0]):
        for j in range(offsprings.shape[1]):
            # Generate a random number between 0 and 1
            random_number = np.random.random()
            # If the random number is less than the mutation rate, flip the bit
            if random_number < mutation_rate:
                offsprings[i, j] = 1 - offsprings[i, j]
    return offsprings


# Define the main Genetic Algorithm function
def genetic_algorithm(
    population_size, chromosome_length, num_generations, mutation_rate
):
    # Initialize the population randomly
    population = np.random.randint(0, 2, size=(population_size, chromosome_length))

    for generation in range(num_generations):
        # Evaluate the fitness of the population
        fitness_values = fitness_function(population)

        # Select the parents for reproduction
        num_parents = population_size // 2
        parents = selection(population, fitness_values, num_parents)

        print(parents, len(parents))

        # Create the offsprings through crossover
        num_offsprings = population_size - num_parents
        offsprings = crossover(parents, num_offsprings)

        print(offsprings)

        # Apply mutation to the offsprings
        offsprings = mutation(offsprings, mutation_rate)

        # Create the new population by combining the parents and offsprings
        population = np.concatenate((parents, offsprings))

    # Return the best individual (solution)
    best_individual_index = np.argmax(fitness_function(population))
    return population[best_individual_index]


# Example usage
population_size = 6
chromosome_length = 5
num_generations = 1
mutation_rate = 0.01

best_individual = genetic_algorithm(
    population_size, chromosome_length, num_generations, mutation_rate
)
print("Best individual:", best_individual)
