import os

from manim import *
import numpy as np


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


# Variables
population_size = 6
chromosome_length = 5


class GeneticAlgorithm(Scene):
    def construct(self):
        title = Tex(r"Genetic Algorithm - OneMax Problem")
        base = Square(color=BLUE)

        VGroup(title, base).arrange(DOWN)
        self.play(Write(title), FadeIn(base, shift=DOWN))
        self.wait()

        self.play(Indicate(base))

        transform_title = Text("One Max Problem")
        transform_title.to_corner(UP + LEFT)

        self.play(
            Transform(title, transform_title),
            LaggedStart(*[FadeOut(obj, shift=DOWN) for obj in base]),
        )

        self.wait()

        # Draw five squares next to each other
        x = (
            VGroup(*[Square() for _ in range(chromosome_length)])
            .set_x(0)
            .arrange(buff=1.0)
        )

        # Write numbers inside each square
        numbers = [Text("1") for _ in range(chromosome_length)]
        for square, number in zip(x, numbers):
            number.move_to(square)

        self.play(FadeIn(x), *[Write(number) for number in numbers])

        self.wait()

        self.play(FadeOut(x), *[FadeOut(number) for number in numbers])

        # Here we draw the squares corresponding to the population
        population = VGroup(
            *[
                VGroup(
                    *[Square(side_length=0.8) for _ in range(chromosome_length)]
                ).arrange(buff=0.5)
                for _ in range(population_size)
            ]
        ).arrange(DOWN)

        # Here we generate random numbers for each square
        numbers = np.random.randint(2, size=(population_size, chromosome_length))

        for i, row in enumerate(population):
            for j, square in enumerate(row):
                number = Text(str(numbers[i][j]))
                number.move_to(square)
                population[i][j].add(number)

        self.play(FadeIn(population))

        fitness_values = fitness_function(numbers)

        fitness_labels = [
            Text("Fitness: " + str(fitness)) for fitness in fitness_values
        ]
        for group, label in zip(population, fitness_labels):
            label.next_to(group, RIGHT)
            self.play(Write(label))
        self.wait()

        self.play(FadeOut(population), *[FadeOut(label) for label in fitness_labels])

        # Select the parents for reproduction
        num_parents = population_size // 2

        parents = selection(numbers, fitness_values, num_parents)
        print(len(parents))

        # Show the parents
        parents_group = VGroup()
        for i, parent in enumerate(parents):
            parent_squares = VGroup(
                *[Square() for _ in range(chromosome_length)]
            ).arrange(buff=0.5)
            parent_numbers = [Text(str(int(value))) for value in parent]
            for square, number in zip(parent_squares, parent_numbers):
                number.move_to(square)
                parent_squares.add(square)
                parent_squares.add(number)
            parents_group.add(parent_squares)
        parents_group.arrange(DOWN)
        self.play(FadeIn(parents_group))
        self.wait()

        self.play(FadeOut(parents_group))

        num_offsprings = 5 - num_parents
        offsprings = np.empty((num_offsprings, parents.shape[1]))

        # TODO change to num_offsprings
        for i in range(1):
            # Select two random parents
            random_indices = np.random.randint(0, parents.shape[0], size=2)
            parent1, parent2 = parents[random_indices]

            # Animate both parents
            parents_group = VGroup()
            for i, parent in enumerate([parent1, parent2]):
                parent_squares = VGroup(
                    *[Square() for _ in range(chromosome_length)]
                ).arrange(buff=0.5)
                parent_numbers = [Text(str(int(value))) for value in parent]
                for square, number in zip(parent_squares, parent_numbers):
                    number.move_to(square)
                    parent_squares.add(square)  # Agrega solo el Square a parent_squares
                parent_group = VGroup(
                    parent_squares, *parent_numbers
                )  # Crea un nuevo VGroup para cada parent
                parents_group.add(parent_group)  # Agrega el VGroup a parents_group
            parents_group.arrange(DOWN)

            self.play(FadeIn(parents_group))
            self.wait()

            crossover_point = 3  # TODO: np.random.randint(0, parents.shape[1])
            print(crossover_point)

            # Animate the crossover point with a Line
            start = parents_group[0][crossover_point].get_corner(UP + RIGHT)
            end = parents_group[1][crossover_point].get_corner(DOWN + RIGHT)

            line = Line(start, end, color=RED, stroke_width=10)
            line.shift(RIGHT * 1.2)  # Move the line a bit to the right
            self.play(Create(line))
            self.wait()

            # Create the offspring by combining the parents' genes
            offspring = np.concatenate(
                (parent1[:crossover_point], parent2[crossover_point:])
            )

            # Apply color change to squares in parents_group[0] and parents_group[1] simultaneously
            self.play(
                *[
                    square.animate.set_color(RED)
                    for square in parents_group[0][0][:crossover_point]
                ]
                + [
                    square.animate.set_color(RED)
                    for square in parents_group[1][0][crossover_point:]
                ]
            )

            self.play(FadeOut(line))

            # Animate the offspring
            # Shrink and move the parents_group
            self.play(parents_group.animate.scale(0.2).to_edge(UP, buff=1))

            self.wait()

            # Create the offspring group

            offsprings[i] = offspring

        print(offsprings)
        print("====")
        print(np.concatenate((parents, offsprings)))


if __name__ == "__main__":
    module_name = os.path.basename(__file__)
    command = "manim -p -ql " + module_name + " GeneticAlgorithm"
    os.system(command)
