import os
import numpy as np

from manim import *

generations = 1
population_size = 6
chromosome_length = 5
tournament_size = population_size // 2

rng = np.random.default_rng()  # TODO: Change seed


def fitness_function(population):
    return np.sum(population, axis=1)


def generate_population(population_size, chromosome_length):
    return rng.integers(0, 2, size=(population_size, chromosome_length))


def selection(population, num_parents):
    tournament = rng.choice(population, num_parents)
    parent1, parent2 = tournament[np.argsort(fitness_function(tournament))[-2:]]
    return parent1, parent2, tournament


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
            # Here we draw the squares corresponding to the population
            population_squares = VGroup(
                *[
                    VGroup(
                        *[Square(side_length=0.8) for _ in range(chromosome_length)]
                    ).arrange(buff=0.5)
                    for _ in range(population_size)
                ]
            ).arrange(DOWN * 0.38)

            # Here we generate random numbers for each square
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

            # Here we generate random numbers for each square
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
                    VGroup(
                        *[Square(side_length=0.8) for _ in range(chromosome_length)]
                    ).arrange(buff=0.5)
                    for _ in range(2)  # Since there are only two parents
                ]
            ).arrange(DOWN)

            print(parent1, parent2)

            self.play(FadeIn(parent_squares))
            self.wait()


if __name__ == "__main__":
    module_name = os.path.basename(__file__)
    command = "manim -p -ql " + module_name + " GeneticAlgorithm"
    os.system(command)
