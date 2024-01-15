import collections
from manim import *


class BFS(Scene):
    def construct(self):
        # Define the vertices
        vertices = [i for i in range(5)]
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (3, 2)]
        root = 0

        # Create the graph
        graph = Graph(
            vertices,
            edges,
            layout="circular",
            layout_scale=3,
            labels=True,
        )

        self.play(Create(graph))
        self.wait()
        self.play(FadeOut(graph))

        # TODO: remove this

        # Create a table and then join them with a VGroup
        table = Table(
            [
                [str(i) for i in range(len(vertices))],
                ["x", "x", "x", "x", "x"],
            ],
            row_labels=[Text("Visited"), Text("Queue")],
            include_outer_lines=True,
            arrange_in_grid_config={"cell_alignment": RIGHT},
        )
        table.get_rows()[0][1:].set_opacity(0)
        table.get_rows()[1][1:].set_opacity(0)

        table.scale(0.5)
        graph.scale(0.6)

        group = VGroup(graph, table)
        group.arrange(RIGHT)
        group.to_edge(ORIGIN)
        self.play(Create(group))

        self.wait()

        # BFS logic and stuff+
        graph_dict = {}
        for edge in edges:
            if edge[0] in graph:
                graph_dict[edge[0]].append(edge[1])
            else:
                graph_dict[edge[0]] = [edge[1]]

        visited, queue = set(), collections.deque([root])
        visited.add(root)

        print(graph_dict)

        self.wait()

        old_entry = table.get_entries((1, 2))[0]
        old_position = old_entry.get_center()

        new_entry = Tex(str(root)).scale(0.5)
        new_entry.move_to(old_position)

        table.get_entries((1, 2))[0] = new_entry

        table.get_entries((1, 2)).set_opacity(1)
        self.wait()

        self.wait()
