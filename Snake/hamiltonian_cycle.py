# https://github.com/maros-o/hamiltonian-cycle
import copy
import random

from gameModule import WALL_CHAR, EMPTY_CHAR


class Vertex:
    def __init__(self, pos, number):
        self.pos = pos
        self.number = number

    def get_pos(self):
        return self.pos

    def get_number(self):
        return self.number


class HamiltonianGenerator:
    def __init__(self, X, Y, grid):
        self.X = X
        self.Y = Y
        self.HALF_X = X // 2
        self.HALF_Y = Y // 2
        self.grid = grid

    def create_nodes(self):
        nodes = [[Vertex((x * 2 + 1, y * 2 + 1), x + y * self.HALF_X) for y in range(0, self.HALF_Y)] for x in
                 range(0, self.HALF_X)]
        return nodes

    def create_edges(self):
        edges = [[0 for y in range(0, self.HALF_Y * self.HALF_X)] for x in range(0, self.HALF_X * self.HALF_Y)]

        skiplist = [self.HALF_X * x for x in range(0, self.HALF_X)]
        for x in range(0, self.HALF_X * self.HALF_Y):
            for y in range(0, self.HALF_Y * self.HALF_X):
                if not (x == y):
                    if (x + 1 == y and y not in skiplist):
                        edges[x][y] = random.randint(1, 3)
                    elif (x + self.HALF_X == y):
                        edges[x][y] = random.randint(1, 3)

        return edges

    def hamiltonian_cycle(self, nodes, edges):
        points = []
        for edge in edges:
            for pos_x in range(0, self.HALF_X):
                for pos_y in range(0, self.HALF_Y):
                    if (nodes[pos_x][pos_y].get_number() == edge[0][0]):
                        start = nodes[pos_x][pos_y].get_pos()
                    if (nodes[pos_x][pos_y].get_number() == edge[0][1]):
                        end = nodes[pos_x][pos_y].get_pos()
            points.append(start)
            points.append(((start[0] + end[0]) // 2, (start[1] + end[1]) // 2))
            points.append(end)

        cycle = [(0, 0)]

        curr = cycle[0]
        dir = (1, 0)

        while len(cycle) < self.X * self.Y:
            x = curr[0]
            y = curr[1]

            if dir == (1, 0):  # right
                if ((x + dir[0], y + dir[1] + 1) in points and (x + 1, y) not in points):
                    curr = (x + dir[0], y + dir[1])
                else:
                    if ((x, y + 1) in points and (x + 1, y + 1) not in points):
                        dir = (0, 1)
                    else:
                        dir = (0, -1)

            elif dir == (0, 1):  # down
                if ((x + dir[0], y + dir[1]) in points and (x + dir[0] + 1, y + dir[1]) not in points):
                    curr = (x + dir[0], y + dir[1])
                else:
                    if ((x, y + 1) in points and (x + 1, y + 1) in points):
                        dir = (1, 0)
                    else:
                        dir = (-1, 0)

            elif dir == (-1, 0):  # left
                if ((x, y) in points and (x, y + 1) not in points):
                    curr = (x + dir[0], y + dir[1])
                else:
                    if ((x, y + 1) not in points):
                        dir = (0, -1)
                    else:
                        dir = (0, 1)

            elif dir == (0, -1):  # up
                if ((x, y) not in points and (x + 1, y) in points):
                    curr = (x + dir[0], y + dir[1])
                else:
                    if ((x + 1, y) in points):
                        dir = (-1, 0)
                    else:
                        dir = (1, 0)

            if curr not in cycle:
                cycle.append(curr)

        return points, cycle

    def prims_algoritm(self, edges):
        clean_edges = []
        for x in range(0, self.HALF_X * self.HALF_Y):
            for y in range(0, self.HALF_Y * self.HALF_X):
                if not (edges[x][y] == 0):
                    clean_edges.append(((x, y), edges[x][y]))

        visited = []
        unvisited = [x for x in range(self.HALF_X * self.HALF_Y)]
        curr = 0

        final_edges = []
        while len(unvisited) > 0:
            visited.append(curr)

            for number in unvisited:
                if number in visited:
                    unvisited.remove(number)

            my_edges = []
            for edge in clean_edges:
                if ((edge[0][0] in visited or edge[0][1] in visited) and not (
                        edge[0][0] in visited and edge[0][1] in visited)):
                    my_edges.append(edge)

            min_edge = ((-1, -1), 999)

            for edge in my_edges:
                if edge[1] < min_edge[1]:
                    min_edge = edge

            if len(unvisited) == 0:
                break

            final_edges.append(min_edge)

            if min_edge[0][0] == -1:
                curr = unvisited[0]
            else:
                if (min_edge[0][1] in visited):
                    curr = min_edge[0][0]
                else:
                    curr = min_edge[0][1]

        return final_edges

    def is_in_grid(self, pos, grid):
        return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])

    def hamiltionian_generate(self):
        graph = []
        wall_list = []
        size = len(self.grid)
        for y in range(size):
            for x in range(size):
                connections = []
                for i in range(size ** 2):
                    connections.append(0)
                if self.grid[y][x] != WALL_CHAR:
                    if self.is_in_grid((x - 1, y), self.grid):
                        connections[x - 1 + y * size] = 1
                    if self.is_in_grid((x + 1, y), self.grid):
                        connections[x + 1 + y * size] = 1
                    if self.is_in_grid((x, y - 1), self.grid):
                        connections[x + (y - 1) * size] = 1
                    if self.is_in_grid((x, y + 1), self.grid):
                        connections[x + (y + 1) * size] = 1
                    graph.append(connections)
                else:
                    wall_list.append(x + y * size)

        counter = 0
        for i in wall_list:
            for row in graph:
                del row[i - counter]
            counter += 1

        NODE = len(graph)
        path = [None] * NODE

        def isValid(v, k):
            if graph[path[k - 1]][v] == 0:  # if there is no edge
                return False

            for i in range(k):  # if vertex is already taken, skip that
                if path[i] == v:
                    return False
            return True

        def cycleFound(k):
            if k == NODE:  # when all vertices are in the path
                if graph[path[k - 1]][path[0]] == 1:
                    return True
                else:
                    return False

            for v in range(1, NODE):  # for all vertices except starting point
                if isValid(v, k):  # if possible to add v in the path
                    path[k] = v
                    if cycleFound(k + 1) == True:
                        return True
                    path[k] = -1  # when k vertex will not be in the solution
            return False

        def hamiltonianCycle():
            for i in range(NODE):
                path[i] = -1
            path[0] = 0  # first vertex as 0

            if cycleFound(1) == False:
                return []

            ham = []
            new_grid = {}
            counter = 0
            for row in range(size):
                for column in range(size):
                    if self.grid[row][column] != WALL_CHAR:
                        new_grid[counter] = (row, column)
                        counter += 1
            for i in path:
                ham.append(new_grid[i])
            return ham

        return hamiltonianCycle()

    def odd_generation(self):
        hamiltonian_path = [[-1 for i in range(self.X)] for j in range(self.Y)]
        hamiltonian_path[1][0] = self.X ** 2 - 3
        hamiltonian_path[1][1] = self.X ** 2 - 2
        for i in range(self.X // 2):
            hamiltonian_path[0][i * 2 + 1] = i * 4
            hamiltonian_path[0][i * 2 + 2] = i * 4 + 1
        for i in range(self.X // 2 - 1):
            hamiltonian_path[1][i * 2 + 2] = i * 4 + 2
            hamiltonian_path[1][i * 2 + 3] = i * 4 + 3
        for i in range(self.X):
            hamiltonian_path[i][-1] = hamiltonian_path[0][-1] + i

        counter = hamiltonian_path[self.X - 1][-1] + 1
        for x in range(self.X - 1, 1, -1):
            for y in range(self.X - 1):
                if x % 2 == 0:
                    hamiltonian_path[x][self.Y - 2 - y] = counter
                    counter += 1
                else:
                    hamiltonian_path[x][y] = counter
                    counter += 1
        return hamiltonian_path

    def generate(self):
        for row in self.grid:
            for cell in row:
                if cell == WALL_CHAR:
                    cycle = self.hamiltionian_generate()
                    if cycle:
                        hamiltonian_path = [[-1 for i in range(self.X)] for j in range(self.Y)]
                        for i in range(len(cycle)):
                            hamiltonian_path[cycle[i][0]][cycle[i][1]] = i
                        return hamiltonian_path
                    return cycle

        if self.X % 2 != 1:
            nodes = self.create_nodes()
            edges = self.create_edges()

            final_edges = self.prims_algoritm(edges)
            points, cycle = self.hamiltonian_cycle(nodes, final_edges)
        else:
            return self.odd_generation()

        hamiltonian_path = [[-1 for i in range(self.X)] for j in range(self.Y)]

        for i in range(len(cycle)):
            hamiltonian_path[cycle[i][0]][cycle[i][1]] = i
        return hamiltonian_path

    def regenerate(self, hamiltonian_path):
        temp = hamiltonian_path[0][0]
        hamiltonian_path[0][0] = hamiltonian_path[1][1]
        hamiltonian_path[1][1] = temp
