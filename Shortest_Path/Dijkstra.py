from matplotlib.pyplot import hist
from FileHandler import FileHandler
from View import View
import sys
import time
from Algorithm import Algorithm


class Dijkstra(Algorithm):

    UNDEFINED = -1

    def __init__(self, dataset):
        super().__init__(dataset, False, False)

    def get_vertex_with_minimal_distance(self, distances, remaining_nodes):
        """
        Gives the closest vertex
        """
        min_node = self.UNDEFINED
        min_value = sys.maxsize
        for node in distances:
            if (node not in remaining_nodes):
                continue
            if (distances[node] < min_value):
                min_node = node
                min_value = distances[node]
        return min_node


    def solve(self):
        """
        Solve the problem using Dijkstra algorithm
        """
        start = 0
        goal = 1

        distances = {}
        predecessors = {}
        set_of_vertices = set()

        for v in self.vertices:
            distances[v] = sys.maxsize
            predecessors[v] = self.UNDEFINED
            set_of_vertices.add(v)

        distances[start] = 0

        while (len(set_of_vertices) > 0):
            current = self.get_vertex_with_minimal_distance(distances, set_of_vertices)
            if (current == self.UNDEFINED):
                print("Error undefined")
                return
            if (current == goal):
                self.cost = distances[goal]
                print("Cost of best path : "+str(self.cost))
                break
            set_of_vertices.remove(current)
            edge_history = []
            vertex_history = []
            current_history = []
            if (current != start and current != goal):
                current_history.append((current, self.COLOR_CURRENT))
                vertex_history.append((current, self.COLOR_EXPLORED))
            neighbour = self.get_neighbors(current)
            for n in neighbour[0]:
                if (n not in set_of_vertices):
                    continue
                edge = self.get_edge(current, n)
                potential_distance = distances[current] + edge[2]
                if (potential_distance < distances[n]):
                    distances[n] = potential_distance
                    predecessors[n] = current
                    edge_history.append(((current, n), self.COLOR_EXPLORED))
                    if (n != goal):
                        vertex_history.append((n, self.COLOR_NEIGHBOURED))
            self.history.append((0, current_history))
            self.history.append((1, edge_history))
            self.history.append((2, vertex_history))
        
        if (predecessors[goal] == self.UNDEFINED):
            print("No solution")
        else:
            print("Solution found")
            path = []
            current = goal
            while (current != self.UNDEFINED):
                path = [current] + path
                current = predecessors[current]
            self.path = path
