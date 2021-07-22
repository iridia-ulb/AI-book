from FileHandler import FileHandler
from GUI import GUI

import heapq
import time
import sys
from enum import Enum

import networkx as nx
import copy


class Heuristic(str, Enum):
    MANHATTAN = "Manhattan"
    EUCLIDIAN = "Euclidian"
    CHEBYSHEV = "Chebyshev"
    DIJKSTRA = "Dijkstra"


HEURISTICS = [e.value for e in Heuristic]


class Algorithm:
    """
    Algorithm of A*, it generates the shortest path for the given instance and shows the result using a GUI
    """

    def __init__(self, instance, heuristic, is_bidirectional, logger, is_using_multithreading=False):
        fh = FileHandler(instance)
        self.E, self.V, self.vertices, self.edges, self.edge_index, self.nodes = fh.read()
        # E is total number of edges
        # V is total number of vertices/nodes
        # vertices is a dictionary that contain vertex id as key and a tuple (position x, position y) as value
        # edges is a list of tuples (source, destination, weight, color)
        # edge_index is a dictionary that stores the index of each edges in the list above
        # nodes is a dictionary where we can get direct neighbors of each nodes and the color assigned to each node
        logger.debug("Instance loaded")
        logger.debug("{} nodes = {}".format(self.V, self.vertices))
        logger.debug("{} edges = {}".format(self.E, self.edges))
        logger.debug("Edges index = {}".format(self.edge_index))
        logger.debug("Nodes neighbors = {}".format(self.nodes))

        # Creating graph object
        self.G = nx.Graph()
        edges_mod = [(x[0], x[1], {"weight": x[2], "color":x[3]}) for x in self.edges]
        self.G.add_edges_from(edges_mod)

        # These attributes are used for the graphical aspect of the code.
        self.history = []

        # Path and cost of the best solution found
        self.path = []
        self.cost = 0
        self.logger = logger
        self.is_bidirectional = is_bidirectional
        self.heuristic_type = heuristic
        self.name = "A* {} {}".format(self.heuristic_type, "bidirectionnal" if self.is_bidirectional else "")

    def show(self):
        """
        Launch GUI
        """
        gui = GUI(self.history, self.G, self.vertices, self.is_bidirectional, self.name, self.logger)
        gui.show()

    def run(self):
        """
        Run desired solve function with time measurement
        """
        self.logger.info(self.name)
        start = time.time()

        if(self.is_bidirectional):
            self.solve_bidirectional()
        else:
            self.solve()

        end = time.time()
        self.logger.info("Time elapsed: {:.2f} ms".format((end-start)*1000))

    def solve(self):
        """
        Solve the problem using the unidirectional A*
        """
        start = 0
        goal = 1
        # items in priority queue:
        # (heuristic (to minimize), current node number, [list of nodes] )
        q = [(0, start, [start])]
        heapq.heapify(q)
        g_scores = {start: 0}
        iterations = -1

        while len(q) != 0:
            iterations += 1
            self.history.append(copy.deepcopy(q))
            self.history[-1].sort()
            (_, current_node, list_of_nodes) = heapq.heappop(q)
            self.logger.debug("Iteration {}, current node: {}, list of nodes: {}".format(iterations,
                                                                                         current_node, list_of_nodes))
            if current_node == goal:
                self.path = list_of_nodes
                self.cost = g_scores[current_node]
                break

            # explore neighboors of the current onde
            for n in list(self.G.adj[current_node]):
                weight = self.G.edges[current_node, n]["weight"]
                g = g_scores[current_node] + weight
                f = g + self.heuristic(n, goal)
                # if this node hasnt been visited yet or the travel cost is smaller, add it to priority queue
                if n not in g_scores or g < g_scores[n]:
                    heapq.heappush(q, (f, n, list_of_nodes + [n]))
                    g_scores[n] = g

        if (len(self.path) == 0):
            self.logger.warn("No solution found")
            return
        self.logger.info("Found path to goal with cost {:.2f} in {} iterations".format(self.cost, iterations))
        self.logger.info("Path: {}".format(self.path))

    def solve_bidirectional(self):
        """
        Solve the problem using the bidirectional A* in a sequential way
        """
        start = 0
        goal = 1
        direction = 0  # 0 or 1

        def other(a):
            return int(not a)

        q = [[(0, start, [start])], [(0, goal, [goal])]]
        heapq.heapify(q[0])
        heapq.heapify(q[1])
        g_scores = [{start: 0}, {goal: 0}]
        save_path = [{start: []}, {goal: []}]
        node_goal = [goal, start]

        keep_searching = True
        iterations = -1

        while len(q) != 0 and keep_searching:
            iterations += 1
            self.history.append(copy.deepcopy(q[direction]))
            self.history[-1].sort()
            (_, current_node, list_of_nodes) = heapq.heappop(q[direction])
            self.logger.debug("Iteration {}, current node: {}, direction: {}, list of nodes: {}".format(iterations,
                                                                                                        current_node, direction, list_of_nodes))
            for n in list(self.G.adj[current_node]):
                weight = self.G.edges[current_node, n]["weight"]
                g = g_scores[direction][current_node] + weight
                f = g + self.heuristic(n, node_goal[direction])
                if n not in g_scores[direction] or g < g_scores[direction][n]:
                    if (n in save_path[other(direction)]):
                        merged_path = copy.deepcopy(save_path[other(direction)][n])
                        merged_path.reverse()
                        self.path = list_of_nodes + merged_path
                        self.cost = g + g_scores[other(direction)][n]
                        if self.path[0] == goal:
                            self.path.reverse()
                        self.history.append(copy.deepcopy([(self.cost, goal, self.path)] + q[direction]))
                        keep_searching = False
                        break
                    else:
                        save_path[direction][n] = list_of_nodes + [n]
                        heapq.heappush(q[direction], (f, n, list_of_nodes + [n]))
                        g_scores[direction][n] = g

            direction = other(direction)

        if (len(self.path) == 0):
            self.logger.warn("No solution found")
            return
        self.logger.info("Found path to goal with cost {:.2f} in {} iterations".format(self.cost, iterations))
        self.logger.info("Path: {}".format(self.path))

    def heuristic(self, a, b):
        """
        Return heuristic between node a and b
        """
        node_a = self.vertices[a]
        node_b = self.vertices[b]
        dx = abs(node_a[0] - node_b[0])
        dy = abs(node_a[1] - node_b[1])
        if(self.heuristic_type == Heuristic.MANHATTAN):
            return (dx + dy)/2
        if(self.heuristic_type == Heuristic.EUCLIDIAN):
            return (dx**2 + dy**2)**(1/2)
        if(self.heuristic_type == Heuristic.CHEBYSHEV):
            return max(dx, dy)
        if(self.heuristic_type == Heuristic.DIJKSTRA):
            return 0
        else:
            return (dx + dy)/2
