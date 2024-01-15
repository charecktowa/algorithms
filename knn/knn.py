import numpy as np


from manim import *
from sklearn.datasets import make_blobs


def generate_dataset(samples: int, centers: int, features: int = 2) -> tuple:
    return make_blobs(
        n_samples=samples,
        centers=centers,
        center_box=(1, 4),
        n_features=features,
        random_state=39,
    )


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNearestNeighbor(Scene):
    def construct(self):
        # Add title
        title = Text("K-Nearest Neighbor").scale(1.5).to_corner(UL)
        self.add(title)

        self.play(FadeOut(title))

        ax = Axes(
            x_range=(0, 6),
            y_range=(0, 6),
            x_length=8,
            y_length=5,
        ).add_coordinates()
        self.add(ax)
        self.wait()

        # Dataset
        X, y = generate_dataset(samples=15, centers=3)

        # Load data
        dots = []
        colors = {0: BLUE, 1: ORANGE, 2: GREEN}
        for element, label in zip(X, y):
            color = colors.get(label, WHITE)
            dots.append(Dot(ax.c2p(element[0], element[1]), color=color))

        self.add(*dots)
        self.wait()

        # Generate new random point
        new_point = np.random.default_rng(42).random(2) * 5
        new_dot = Dot(ax.c2p(new_point[0], new_point[1]), color=WHITE)

        self.add(new_dot)
        self.wait()

        # Initialize k
        text = Text("Seleccionamos un valor para k").scale(0.7).to_corner(UL)
        k = 3
        self.add(text, Text(f"k = {k}").to_corner(UR))
        self.wait()
        self.play(FadeOut(text))
        distances = []
        for dot in dots:
            text = (
                Text("Calculamos la distancias entre puntos").scale(0.7).to_corner(UL)
            )
            self.play(FadeIn(text))
            self.play(Indicate(dot), Indicate(new_dot))

            # Draw line between dots
            dash = DashedLine(dot.get_center(), new_dot.get_center(), color=YELLOW_C)
            self.add(dash)
            self.wait()

            # Calculate distances
            distances.append(
                euclidean_distance(
                    np.array([dot.get_center()[0], dot.get_center()[1]]), new_point
                )
            )

            text_distance = Text(f"{distances[-1]:.2f}").scale(0.2)
            text_distance.next_to(dash, UP)

            dot.add(
                text_distance,
                Text(f"Distancia: {distances[-1]:.2f}").scale(0.75).to_corner(DL),
            )
            self.wait()
            dot.remove(dot[-1])
            dot.remove(text_distance)
            dash.set_color(WHITE)
            self.wait()

        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # Show table with distances
        headers = ["Dot", "New Dot", "Distance"]

        # Prepare data for table
        table_data = [headers]
        for i, distance in enumerate(distances):
            dot_label = f"{X[i]}"
            new_dot_label = f"{new_point}"
            distance_text = f"{distance:.2f}"
            table_data.append([dot_label, new_dot_label, distance_text])

        # Create the table

        distance_table = Table(
            table_data,
            include_outer_lines=True,
            arrange_in_grid_config={"cell_alignment": RIGHT},
        )

        # Center the table in the scene
        distance_table.scale_to_fit_height(5.8)
        distance_table.move_to(ORIGIN)

        # Add the table to the scene
        text = Text("Tabla de distancias").scale(0.7).to_corner(UL)
        self.play(FadeIn(text, distance_table))
        self.wait()

        self.play(FadeOut(text, distance_table))

        # Sort distances
        sorted_distances = sorted(distances)

        # Show table with distances
        headers = ["Dot", "New Dot", "Distance"]

        # Prepare data for table
        table_data = [headers]
        for i, distance in enumerate(sorted_distances):
            dot_label = f"{X[i]}"
            new_dot_label = f"{new_point}"
            distance_text = f"{distance:.2f}"
            table_data.append([dot_label, new_dot_label, distance_text])

        # Create the table

        distance_table = Table(
            table_data,
            include_outer_lines=True,
            arrange_in_grid_config={"cell_alignment": RIGHT},
        )

        # Center the table in the scene
        distance_table.scale_to_fit_height(5.8)
        distance_table.move_to(ORIGIN)

        # Add the table to the scene
        text = (
            Text("Ordenamos las distancias de menor a mayor").scale(0.5).to_corner(UL)
        )
        self.play(FadeIn(text, distance_table))
        self.wait()

        self.play(FadeOut(text, distance_table))
        self.wait()

        # Get k nearest neighbors
        # Put the Ax and show a cirlce to put the k nearest neighbors
        self.add(ax)

        circle = DashedVMobject(
            Circle(
                radius=sorted_distances[k - 1] / 2,
                color=RED,
            )
            .move_to(new_dot.get_center())
            .surround(new_dot, buffer_factor=sorted_distances[k - 1] * k)
        )

        text = Text("Tomamos las k distancias más cercanas").scale(0.5).to_corner(UL)
        self.play(Create(circle))
        self.add(new_dot)
        self.play(FadeIn(text, *dots))
        self.add(Text(f"k = {k}").to_corner(UR))
        self.wait()

        # Show k nearest neighbors in the table
        for i in range(k):
            self.play(Indicate(distance_table.get_rows()[i + 1]))

        self.wait()

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        text = (
            Text("Contamos las etiquetas de las k distancias").scale(0.5).to_corner(UL)
        )

        self.play(FadeIn(text))

        headers = ["Dot", "Label", "Unknown Dot", "Distance"]
        table_data = [headers]
        for i, distance in enumerate(distances[:k]):
            dot_label = f"{X[i]}"
            label = f"{y[i]}"
            new_dot_label = f"{new_point}"
            distance_text = f"{distance:.2f}"
            table_data.append([dot_label, label, new_dot_label, distance_text])

        distance_table = Table(
            table_data,
            include_outer_lines=True,
            arrange_in_grid_config={"cell_alignment": RIGHT},
        )

        # Center the table in the scene
        distance_table.scale_to_fit_height(2.8)
        distance_table.move_to(ORIGIN)

        self.play(FadeIn(distance_table))
        self.wait()

        # Get most common label

        ## Highlight label column
        for i in range(k):
            self.play(Indicate(distance_table.get_rows()[i + 1][1]))

        self.wait()

        ## Get most common label
        most_common_label = np.bincount(y[:k]).argmax()

        ## Highlight most common label
        common = (
            Text(f"Etiqueta mayoritaria: {most_common_label}").scale(0.75).to_corner(DL)
        )
        self.play(FadeIn(common))
        self.wait()

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        ## Show ax
        text = (
            Text("Damos la nueva etiqueta al punto desconocido")
            .scale(0.5)
            .to_corner(UL)
        )
        self.play(FadeIn(text, ax))
        self.wait()
        self.play(FadeIn(new_dot))
        self.wait()
        self.play(Indicate(new_dot))
        self.wait()

        # Change new_dot color to corresponding label
        new_dot.set_color(colors[most_common_label])
        self.wait()

        self.play(FadeIn(*dots))
        self.wait()
