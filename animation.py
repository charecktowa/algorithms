from hmac import new
import os
import numpy as np

from manim import *

generations = 5
population_size = 6
chromosome_length = 5
tournament_size = population_size // 2
mutation_rate = 0.05

rng = np.random.default_rng(62)


def fitness_function(population):
    return np.sum(population, axis=1)


def generate_population(population_size, chromosome_length):
    return rng.integers(0, 2, size=(population_size, chromosome_length))


def selection(population, num_parents):
    tournament = rng.choice(population, num_parents)
    parent1, parent2 = tournament[np.argsort(fitness_function(tournament))[-2:]]
    return parent1, parent2, tournament


def mutation(individual, mutation_rate):
    mutated_individual = individual.copy()
    mutated_indexes = []
    for i in range(len(mutated_individual)):
        if rng.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
            mutated_indexes.append(i)
    return mutated_individual, mutated_indexes


class GeneticAlgorithm(Scene):
    def construct(self):
        title = Tex(r"Genetic Algorithm - OneMax Problem")
        base = Square(color=BLUE)

        VGroup(title, base).arrange(DOWN)
        self.play(Write(title), FadeIn(base, shift=DOWN))
        self.play(Indicate(base))
        self.wait()
        self.play(FadeOut(title))

        title = Text("One Max Problem")
        title.to_corner(UP + LEFT)
        self.play(
            Transform(title, title),
            LaggedStart(*[FadeOut(obj, shift=DOWN) for obj in base]),
        )
        self.wait()

        # Draw five squares next to each other
        sample_squares = (
            VGroup(*[Square() for _ in range(5)]).set_x(0).arrange(buff=1.0)
        )

        # Write numbers inside each square
        numbers = [Text("1") for _ in range(5)]
        for square, number in zip(sample_squares, numbers):
            number.move_to(square)

        self.play(FadeIn(sample_squares), *[Write(number) for number in numbers])
        self.wait()
        self.play(FadeOut(sample_squares), *[FadeOut(number) for number in numbers])

        # Here the genetic algorithm starts

        # Initialize the population randomly
        population = generate_population(population_size, chromosome_length)

        # Animation to show the population

        for _ in range(generations):
            new_population = []

            while len(new_population) < population_size:
                # Here we draw the squares corresponding to the population
                population_squares = VGroup(
                    *[
                        VGroup(
                            *[Square(side_length=0.8) for _ in range(chromosome_length)]
                        ).arrange(buff=0.5)
                        for _ in range(population_size)
                    ]
                ).arrange(DOWN * 0.38)

                for i, row in enumerate(population_squares):
                    for j, square in enumerate(row):
                        number = Text(str(population[i][j]))
                        number.move_to(square)
                        population_squares[i][j].add(number)

                self.play(FadeIn(population_squares))
                self.wait()

                # Now we calculate the fitness of each individual
                fitness = fitness_function(population)
                fitness_labels = [Text("Fitness: " + str(fit)) for fit in fitness]
                for group, label in zip(population_squares, fitness_labels):
                    label.next_to(group, RIGHT)
                    self.play(Write(label))
                self.wait()

                self.play(
                    FadeOut(population_squares),
                    *[FadeOut(label) for label in fitness_labels],
                )

                # Select the parents for reproduction
                parent1, parent2, tournament = selection(population, tournament_size)

                # Show who are going to compete
                tournament_squares = VGroup(
                    *[
                        VGroup(
                            *[Square(side_length=0.8) for _ in range(chromosome_length)]
                        ).arrange(buff=0.5)
                        for _ in range(tournament_size)
                    ]
                ).arrange(DOWN)

                for i, row in enumerate(tournament_squares):
                    for j, square in enumerate(row):
                        number = Text(str(tournament[i][j]))
                        number.move_to(square)
                        tournament_squares[i][j].add(number)

                self.play(FadeIn(tournament_squares))
                self.wait()

                # Indicate the parents TODO: if it repeats dont show it again
                for i, individual in enumerate(tournament):
                    if np.array_equal(individual, parent1) or np.array_equal(
                        individual, parent2
                    ):
                        self.play(Indicate(tournament_squares[i]))
                        self.wait()

                self.play(FadeOut(tournament_squares))

                # Show the parents
                parent_squares = VGroup(
                    *[
                        VGroup(*[Square() for _ in range(chromosome_length)]).arrange(
                            buff=0.5
                        )
                        for _ in range(2)  # Since there are only two parents
                    ]
                ).arrange(DOWN)

                parents = [parent1, parent2]
                for i, row in enumerate(parent_squares):
                    for j, square in enumerate(row):
                        number = Text(str(parents[i][j]))
                        number.move_to(square)
                        parent_squares[i][j].add(number)

                self.play(FadeIn(parent_squares))
                self.wait()

                # Perform crossover
                crossover_point = rng.integers(1, chromosome_length)
                child1 = np.concatenate(
                    (parent1[:crossover_point], parent2[crossover_point:])
                )
                child2 = np.concatenate(
                    (parent2[:crossover_point], parent1[crossover_point:])
                )

                # Animate crossover line
                crossover_line = Line(
                    start=parent_squares[0][crossover_point - 1].get_corner(UP + RIGHT),
                    end=parent_squares[1][crossover_point - 1].get_corner(DOWN + RIGHT),
                    color=RED,
                    stroke_width=11,
                )
                crossover_line.shift(RIGHT * 0.2)
                self.play(Create(crossover_line))
                self.wait()

                # Apply color changes to squares in parent_squares
                self.play(
                    *[
                        square.animate.set_color(RED)
                        for square in parent_squares[0][crossover_point:]
                    ]
                    + [
                        square.animate.set_color(RED)
                        for square in parent_squares[1][:crossover_point]
                    ]
                )
                self.wait()

                self.play(FadeOut(crossover_line))

                # Shrink and move the parent_squares
                self.play(parent_squares.animate.scale(0.35).to_edge(UP, buff=1))
                self.wait()

                # Animate the offspring
                offspring_squares = VGroup(
                    *[
                        VGroup(*[Square() for _ in range(chromosome_length)]).arrange(
                            buff=0.5
                        )
                        for _ in range(2)  # Since there are only two offsprings
                    ]
                ).arrange(DOWN)

                offspring = [child1, child2]
                for i, row in enumerate(offspring_squares):
                    for j, square in enumerate(row):
                        number = Text(str(offspring[i][j]))
                        number.move_to(square)
                        offspring_squares[i][j].add(number)

                offspring_squares.scale(0.8).to_edge(DOWN, buff=1)
                self.play(FadeIn(offspring_squares))
                self.wait()

                # Apply color changes to squares in offspring_squares
                self.play(
                    *[
                        square.animate.set_color(BLUE)
                        for square in offspring_squares[0][crossover_point:]
                    ]
                    + [
                        square.animate.set_color(BLUE)
                        for square in offspring_squares[1][:crossover_point]
                    ]
                )
                self.wait()

                # Apply color changes to squares in offspring_squares
                self.play(
                    *[
                        square.animate.set_color(WHITE)
                        for square in offspring_squares[0][crossover_point:]
                    ]
                    + [
                        square.animate.set_color(WHITE)
                        for square in offspring_squares[1][:crossover_point]
                    ]
                )

                child1, mutated1 = mutation(child1, mutation_rate)
                child2, mutated2 = mutation(child2, mutation_rate)

                if mutated1 or mutated2:
                    offspring = [child1, child2]
                    self.play(
                        *[
                            Indicate(offspring_squares[0][index], color=RED)
                            for index in mutated1
                        ]
                        + [
                            Indicate(offspring_squares[1][index], color=RED)
                            for index in mutated2
                        ]
                    )

                    for i, mutated_indices in enumerate([mutated1, mutated2]):
                        for index in mutated_indices:
                            number = Text(str(offspring[i][index]))
                            offspring_squares[i][index].remove(
                                offspring_squares[i][index][1]
                            )
                            number.move_to(offspring_squares[i][index])
                            offspring_squares[i][index].add(number)
                    self.wait()

                # Add the offspring to the new population
                new_population.append(child1)
                new_population.append(child2)

                # Fade out the offspring
                self.play(FadeOut(offspring_squares))
                self.play(*[FadeOut(mob) for mob in self.mobjects])
                self.wait()

            # Update the population
            new_population = np.array(new_population)
            population = new_population

            # Show final population from this generation
            print("printing...")
            population_squares = VGroup(
                *[
                    VGroup(
                        *[Square(side_length=0.85) for _ in range(chromosome_length)]
                    ).arrange(buff=0.5)
                    for _ in range(population_size)
                ]
            ).arrange(DOWN * 0.38)

            for i, row in enumerate(population_squares):
                for j, square in enumerate(row):
                    number = Text(str(population[i][j]))
                    number.move_to(square)
                    population_squares[i][j].add(number)

            self.play(FadeIn(population_squares))
            self.wait()

            self.play(FadeOut(population_squares))


if __name__ == "__main__":
    module_name = os.path.basename(__file__)
    command = "manim -p -ql " + module_name + " GeneticAlgorithm"
    os.system(command)
