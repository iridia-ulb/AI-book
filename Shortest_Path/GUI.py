import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

COLOR_NEIGHBOURED = "#85fc23"
COLOR_EXPLORED = "#65d6d2"
COLOR_SRC = "green"
COLOR_DST = "red"
COLOR_PATH = "#ffb700"
COLOR_BIDIRECTIONAL = "#a834eb"
COLOR_CURRENT = "#FF33ff"  # "#ff9d00"
COLOR_SHORT_LIST = "#ffff00"


class GUI:
    """
    Class used for the GUI of the project
    """

    def __init__(
        self, history, graph, nodes_pos, is_bidirectional, name, logger
    ):
        self.history = history
        self.is_bidirectional = is_bidirectional
        self.G = graph
        self.nodes_pos = nodes_pos
        self.nodes_pos_without_goals = b = {
            x: self.nodes_pos[x] for x in self.nodes_pos if x not in [0, 1]
        }
        self.name = name
        self.logger = logger
        self.index = 0

        self.edges_index = list(self.G.edges())
        self.node_labels = {x: "{}".format(x) for x in list(self.G.nodes())}
        self.node_labels[0] = "start"
        self.node_labels[1] = "goal"

    def plotOneState(self, state, iteration, revState=[]):

        # revstate is filled only when bidirectionnal search is performed
        candidates = state[1:] + revState

        # Set title, texts and legends
        plt.clf()
        plt.cla()
        ax = plt.gca()
        ax.set_title(f"{self.name}: iteration {iteration}")
        # Compute current path cost
        (score, _, path) = state[0]
        path_cost = 0
        for i, k in zip(path[0::1], path[1::1]):
            path_cost += self.G.edges[i, k]["weight"]
        heuristic_cost = score - path_cost
        final = path[-1] == 1 and path[0] == 0

        plt.gcf().text(
            0.3,
            0.15,
            f"Heuristic + path cost = {heuristic_cost:.2f} + {path_cost:.2f} = {score:.2f}",
            fontsize=11,
        )

        legend_list = [
            ("Current", COLOR_CURRENT),
            ("Explored", COLOR_EXPLORED),
            ("Open list", COLOR_NEIGHBOURED),
            ("Short list", COLOR_SHORT_LIST),
            ("Path", COLOR_PATH),
            ("Final path", COLOR_DST),
            ("Current of other branch", COLOR_BIDIRECTIONAL),
        ]
        if self.is_bidirectional:
            x_pos = 0.05
        else:
            x_pos = 0.25
        y_pos = 0.1
        x_step = 0.13
        for (text, col) in legend_list:
            if not self.is_bidirectional and "other" in text:
                continue
            if not final and "Final" in text:
                continue
            if final and "Path" in text:
                continue
            plt.gcf().text(
                x_pos, y_pos, text, color="k", fontsize=10, backgroundcolor=col
            )
            x_pos += x_step

        # Fill nodes and edges color wrt current state
        edges_color = ["black" for _ in self.G.edges()]
        nodes_color = ["grey" for _ in self.G.nodes()]

        # candidates nodes in blue
        for (nscore, n, path) in candidates:
            for i in path:
                nodes_color[i] = COLOR_SHORT_LIST
            for i, k in zip(path[0::1], path[1::1]):
                try:
                    edges_color[self.edges_index.index((i, k))] = COLOR_EXPLORED
                except ValueError:
                    edges_color[self.edges_index.index((k, i))] = COLOR_EXPLORED
            nodes_color[n] = COLOR_NEIGHBOURED

        # current
        current_colors = [COLOR_CURRENT, COLOR_BIDIRECTIONAL]
        for ix, current_state in enumerate(state[:1] + revState[:1]):
            (score, current_node, path) = current_state
            for i in path:
                nodes_color[i] = COLOR_DST if final else COLOR_PATH
            for i, k in zip(path[0::1], path[1::1]):
                try:
                    edges_color[self.edges_index.index((i, k))] = (
                        COLOR_DST if final else COLOR_PATH
                    )
                except ValueError:
                    edges_color[self.edges_index.index((k, i))] = (
                        COLOR_DST if final else COLOR_PATH
                    )
            nodes_color[current_node] = (
                COLOR_DST if final else current_colors[ix]
            )

        # First draw start and goal nodes with star shape
        nx.draw_networkx_nodes(
            self.G,
            self.nodes_pos,
            nodelist=[0, 1],
            node_color=nodes_color[:2],
            node_shape="*",
            node_size=400,
        )
        # Draw all nodes and edges
        nx.draw(
            self.G,
            self.nodes_pos,
            with_labels=True,
            labels=self.node_labels,
            font_size=10,
            node_color=nodes_color[2:],
            nodelist=self.nodes_pos_without_goals,
            edge_color=edges_color,
        )
        # Add edges labels
        nx.draw_networkx_edge_labels(
            self.G,
            self.nodes_pos,
            edge_labels=nx.get_edge_attributes(self.G, "weight"),
            font_size=6,
        )

        plt.subplots_adjust(bottom=0.2)

    def show(self):
        self.showButtons()

    def plotIndex(self):
        if self.is_bidirectional:
            other_index = (
                self.index + 1
                if self.index + 1 < len(self.history)
                else self.index - 1
            )
            iteration = self.index - 1 if self.index > 0 else 0
            self.plotOneState(
                self.history[self.index],
                iteration=iteration,
                revState=self.history[other_index],
            )
        else:
            self.plotOneState(self.history[self.index], iteration=self.index)

    def showButtons(self):
        def next(event):
            self.index += 1
            self.index = min(self.index, len(self.history) - 1)
            self.showButtons()

        def back(event):
            self.index -= 1
            self.index = max(self.index, 0)
            self.showButtons()

        def end(event):
            self.index = len(self.history) - 1
            self.showButtons()

        def reset(event):
            self.index = 0
            self.showButtons()

        self.plotIndex()
        self.button_reset = widgets.Button(
            plt.axes([0.5, 0.01, 0.1, 0.05]), "Reset"
        )
        self.button_back = widgets.Button(
            plt.axes([0.6, 0.01, 0.1, 0.05]), "Back"
        )
        self.button_next = widgets.Button(
            plt.axes([0.7, 0.01, 0.1, 0.05]), "Next"
        )
        self.button_end = widgets.Button(
            plt.axes([0.8, 0.01, 0.1, 0.05]), "End"
        )

        self.button_next.on_clicked(next)
        self.button_back.on_clicked(back)
        self.button_end.on_clicked(end)
        self.button_reset.on_clicked(reset)
        plt.show()
