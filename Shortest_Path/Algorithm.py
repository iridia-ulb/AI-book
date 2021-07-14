from matplotlib.pyplot import hist
from FileHandler import FileHandler
from View import View
import heapq

import threading
import time


class Algorithm:
    """
    Algorithm of A*, it generates the shortest path for the given instance and shows the result using a GUI 
    """

    COLOR_NEIGHBOURED = "#85fc23"
    COLOR_EXPLORED = "#65d6d2"
    COLOR_SRC = "green"
    COLOR_DST = "red"
    COLOR_PATH = "#ffb700"
    COLOR_BIDIRECTIONAL = "#a834eb"
    COLOR_CURRENT = "#ff9d00"



    def __init__(self, dataset, is_bidirectional, is_using_multithreading = False):
        fh = FileHandler(dataset)
        self.E, self.V, self.vertices, self.edges, self.edge_index, self.nodes = fh.read()
        # E is total number of edges
        # V is total number of vertices
        # vertices is a dictionary that contain vertex id as key and a tuple (position x, position y) as value
        # edges is a list of tuples (source, destination, weight, color)
        # edge_index is a dictionary that stores the index of each edges in the list above
        # nodes is a dictionary where we can get direct neighbors of each nodes and the color assigned to each node


        # if you want to solve before visualizing the process you can do it here, or you can solve during the
        # visualization in the update function

        # These attributes are used for the graphical aspect of the code.
        self.counter_history = 0
        self.history = []
        self.iteration = 0


        # Path and cost of the best solution found
        self.path = []
        self.cost = 0

        start = time.time()

        # Check wheter it has to use the bidirectional version or the unidirectional one
        # If the bidirectional version is chosen, it checks if it has to use multithreading or not

        if(not is_bidirectional):
            self.solve()
        else:
            if (is_using_multithreading):
                self.solve_bidirectional_multi()
            else:
                self.solve_bidirectional()

        end = time.time()
        print("Time elapsed (ms) : "+str(int((end-start)*1000*1000)/1000))

        # Once the algorithm has solved the problem, we removed the irrelevant moves from the history (graphical aspect only)
        self.remove_bad_history()


        # Generate the GUI
        self.view = View(self, label_edges=True, speed=500)

    def remove_bad_history(self):
        """
        Remove the irrelevant moves from the history (graphical aspect only)
        """
        real_history = []
        for h in self.history:
            if (len(h[1]) > 0):
                real_history.append(h)
        self.history = real_history 

    def get_remaining_counter(self):
        """
        Gives the number of times the counter_history can increment before reaching the end (graphical aspect only)
        """
        return len(self.history) - self.counter_history + 1

    def get_length_history(self):
        """
        Get the length of the history (graphical aspect only)
        """
        return len(self.history)
    
    def get_counter_history(self):
        """
        Get the counter_history (graphical aspect only)
        """
        return self.counter_history

    def reinitialize_history(self, data):
        """
        Reset the animation by setting the counters to 0 and reinitializing the colors (graphical aspect only)
        """
        self.iteration = 0
        self.counter_history = 0
        edges, nodes = data
        self.edges = edges
        self.nodes = nodes
    
    


    def update(self):
        """
        Update for the animation, following the history of the execution (graphical aspect only)
        return the current counter_history
        """
        if (self.counter_history > len(self.history)):
            return self.counter_history
        if (self.counter_history == len(self.history)):
            for i in range(len(self.path)-2):
                src = self.path[i]
                dst = self.path[i+1]
                self.set_edge_color(src, dst, self.COLOR_PATH)
                self.set_node_color(dst,  self.COLOR_PATH)
            self.set_edge_color(self.path[len(self.path)-2], self.path[len(self.path)-1],  self.COLOR_PATH)
            self.counter_history += 1
            return self.counter_history
        current_history = self.history[self.counter_history]
        if (current_history[0] == 0): #current
            self.iteration += 1
            current_history = current_history[1]
            for n in current_history:
                self.set_node_color(n[0], n[1])
        elif (current_history[0] == 1): #edges animation
            current_history = current_history[1]
            for e in current_history:
                self.set_edge_color(e[0][0], e[0][1], e[1])
        else: #Vertex animation
            current_history = current_history[1]
            for n in current_history:
                self.set_node_color(n[0], n[1])
            
        self.counter_history +=1
        return self.counter_history

    def solve(self):
        """
        Solve the problem using the unidirectional A*
        """
        start = 0
        goal = 1
        q = [(0, start, [start])]
        heapq.heapify(q)

        self.history = []

        self.path = []

        g_scores = {start: 0}
        while len(q) != 0:
            current = heapq.heappop(q)
            if current[1] == goal:
                self.path = current[2]
                self.cost = g_scores[current[1]]
                print("Cost of best path : "+str(self.cost))
                break
            # These three variables are used for the history
            edge_history = []
            vertex_history = []
            current_history = []
            if (current[1] != start and current[1] != goal):
                current_history.append((current[1], self.COLOR_CURRENT))
                vertex_history.append((current[1], self.COLOR_EXPLORED))
            neighbour = self.get_neighbors(current[1])
            for n in neighbour[0]:
                edge = self.get_edge(current[1], n)
                g = g_scores[current[1]] + edge[2] #weight
                f = g + self.heuristic(n, goal)
                if n not in g_scores or g < g_scores[n]:
                    heapq.heappush(q, (f, n, current[2] + [n]))
                    edge_history.append(((current[1], n), self.COLOR_EXPLORED))
                    if (n != goal):
                        vertex_history.append((n, self.COLOR_NEIGHBOURED))
                    g_scores[n] = g
            self.history.append((0, current_history))
            self.history.append((1, edge_history))
            self.history.append((2, vertex_history))
        if (len(self.path) == 0) :
            print("No solution")
        else:
            print("Solution found")

    def solve_bidirectional(self):
        """
        Solve the problem using the bidirectional A* in a sequential way
        """
        start = 0
        goal = 1
        q1 = [(0, start, [start])]
        q2 = [(0, goal, [goal])]
        heapq.heapify(q1)
        heapq.heapify(q2)


        self.history = []

        self.path = []

        g_scores_1 = {start: 0}
        g_scores_2 = {goal: 0}
        save_path_1 = {start : []}
        save_path_2 = {goal : []}


        is_first = True
        q = q1

        while len(q) != 0:
            if (is_first):
                q = q1
                g_scores = g_scores_1
                g_scores_other = g_scores_2
                save_path = save_path_1
                save_path_other = save_path_2
                to_reach = goal
            else:
                q = q2
                g_scores = g_scores_2
                g_scores_other = g_scores_1
                save_path = save_path_2
                save_path_other = save_path_1
                to_reach = start
            current = heapq.heappop(q)
            if current[1] in save_path_other:
                save_path_other[current[1]].reverse()
                self.path = current[2] + save_path_other[current[1]]
                self.cost = g_scores[current[1]] + g_scores_other[current[1]]
                print("Cost of best path : "+str(self.cost))
                break
            edge_history = []
            vertex_history = []
            current_history = []
            if (current[1] != start and current[1] != goal):
                current_history.append((current[1], self.COLOR_CURRENT))
                vertex_history.append((current[1], self.COLOR_EXPLORED))
            neighbour = self.get_neighbors(current[1])
            for n in neighbour[0]:
                edge = self.get_edge(current[1], n)
                g = g_scores[current[1]] + edge[2] #weight
                f = g + self.heuristic(n, goal)
                if n not in g_scores or g < g_scores[n]:
                    save_path[n] = current[2] + [n]
                    heapq.heappush(q, (f, n, current[2] + [n]))
                    edge_history.append(((current[1], n), self.COLOR_EXPLORED))
                    
                    g_scores[n] = g
                    if (n in save_path_other):
                        save_path_other[n].reverse()
                        self.path = current[2] + save_path_other[n]
                        self.cost = g_scores[n] + g_scores_other[n]
                        vertex_history.append((n, self.COLOR_BIDIRECTIONAL))
                        print("Cost of best path : "+str(self.cost))
                        self.history.append((0, current_history))
                        self.history.append((1, edge_history))
                        self.history.append((2, vertex_history))
                        print("Solution found")
                        return
                    if (n != to_reach):
                        vertex_history.append((n, self.COLOR_NEIGHBOURED))
            self.history.append((0, current_history))
            self.history.append((1, edge_history))
            self.history.append((2, vertex_history))
            is_first = not is_first
        if (len(self.path) == 0) :
            print("No solution")
        else :
            print("Solution found")

    def solve_bidirectional_multi(self):
        """
        Solve the problem using the bidirectional A* with multithreading
        """

        mutex_already_found = threading.Lock()
        mutex_history = threading.Lock()   
        self.already_found = False

        def processData(start, goal, q, g_scores, g_scores_other, save_path, save_path_other, mutex_path, mutex_path_other):
            counter = 0
            while (len(q) != 0 and not self.already_found):
                counter += 1
                current = heapq.heappop(q)
                if current[1] in save_path_other:
                    mutex_already_found.acquire()
                    if (not self.already_found):
                        self.already_found = True
                        save_path_other[current[1]].reverse()
                        self.path = current[2] + save_path_other[current[1]]
                        self.cost = g_scores[current[1]] + g_scores_other[current[1]]
                        print("Cost of best path : "+str(self.cost))
                    mutex_already_found.release()
                    break
                edge_history = []
                vertex_history = []
                current_history = []
                if (current[1] != start and current[1] != goal):
                    current_history.append((current[1], self.COLOR_CURRENT))
                    vertex_history.append((current[1], self.COLOR_EXPLORED))
                neighbour = self.get_neighbors(current[1])
                for n in neighbour[0]:
                    edge = self.get_edge(current[1], n)
                    g = g_scores[current[1]] + edge[2] #weight
                    f = g + self.heuristic(n, goal)
                    if n not in g_scores or g < g_scores[n]:
                        mutex_path.acquire()
                        save_path[n] = current[2] + [n]
                        mutex_path.release()
                        heapq.heappush(q, (f, n, current[2] + [n]))
                        edge_history.append(((current[1], n), self.COLOR_EXPLORED))
                        
                        g_scores[n] = g
                        mutex_path_other.acquire()
                        if (n in save_path_other):
                            mutex_already_found.acquire()
                            if (not self.already_found):
                                self.already_found = True
                                save_path_other[n].reverse()
                                self.path = current[2] + save_path_other[n]
                                self.cost = g_scores[n] + g_scores_other[n]
                                vertex_history.append((n, self.COLOR_BIDIRECTIONAL))
                                print("Cost of best path : "+str(self.cost))
                                mutex_history.acquire()
                                self.history.append((0, current_history))
                                self.history.append((1, edge_history))
                                self.history.append((2, vertex_history))
                                mutex_history.release()
                                print("Solution found")
                            mutex_already_found.release()
                            mutex_path_other.release()
                            return
                        mutex_path_other.release()
                        if (n != goal):
                            vertex_history.append((n, self.COLOR_NEIGHBOURED))
                mutex_history.acquire()
                self.history.append((0, current_history))
                self.history.append((1, edge_history))
                self.history.append((2, vertex_history))
                mutex_history.release()
            

        start = 0
        goal = 1
        q1 = [(0, start, [start])]
        q2 = [(0, goal, [goal])]
        heapq.heapify(q1)
        heapq.heapify(q2)


        self.history = []

        self.path = []

        g_scores_1 = {start: 0}
        g_scores_2 = {goal: 0}
        save_path_1 = {start : []}
        save_path_2 = {goal : []}

        mutex_path_1 = threading.Lock()
        mutex_path_2 = threading.Lock()

        t1 = threading.Thread(target = processData, args=(start, goal, q1, g_scores_1, g_scores_2, save_path_1, save_path_2, mutex_path_1, mutex_path_2))
        t2 = threading.Thread(target = processData, args=(goal, start, q2, g_scores_2, g_scores_1, save_path_2, save_path_1, mutex_path_2, mutex_path_1))
        t1.start()
        t2.start()

        t1.join()
        t2.join()

        
    def get_nodes(self):
        return self.nodes

    def get_iteration(self):
        return self.iteration

    def get_edges(self):
        return self.edges

    def set_edge_color(self, src, dest, color):
        """
        Change the color of the given edge
        """
        if (src, dest) in self.edge_index:
            index = self.edge_index[(src, dest)]
            self.edges[index] = self.edges[index][0], self.edges[index][1], self.edges[index][2], color
        elif (dest, src) in self.edge_index:
            index = self.edge_index[(dest, src)]
            self.edges[index] = self.edges[index][0], self.edges[index][1], self.edges[index][2], color

    def set_node_color(self, node_id, color):
        """
        Change the color of the given node
        """
        self.nodes[node_id] = (self.nodes[node_id][0], color)

    def get_edge(self, a, b):
        if (a, b) in self.edge_index:
            index = self.edge_index[(a, b)]
            return self.edges[index]
        elif (b, a) in self.edge_index:
            index = self.edge_index[(b, a)]
            return self.edges[index]

    def get_neighbors(self, v):
        """
        Gives the neighbours of node v
        """
        return self.nodes[v]

    def get_vertices(self):
        return self.vertices


    def get_vertex(self, id):
        return self.vertices[id]

        


    

    def heuristic(self, a, b):
        return 0

