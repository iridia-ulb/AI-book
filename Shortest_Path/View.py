import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import time
import threading
import copy


class View:
    """
    Class used for the GUI of the project
    """

    def __init__(self, algorithm, label_edges=True, speed=1000):
        self.algorithm = algorithm
        self.label_edges = label_edges
        self.speed = speed
        self.fig, self.ax = plt.subplots()
        self.G = nx.Graph()
        for v in range(len(self.algorithm.get_vertices())):
            self.G.add_node(v, pos=self.algorithm.get_vertices()[v])
        for e in self.algorithm.get_edges():
            self.G.add_edge(e[0], e[1], color=e[3], weight=e[2])
        if (len(self.algorithm.get_vertices()) <= 100):
            self.pos = nx.spring_layout(self.G, pos=self.algorithm.get_vertices(), fixed=self.algorithm.get_vertices())
        
        self.save_init_config = (copy.deepcopy(self.algorithm.get_edges()), copy.deepcopy(self.algorithm.get_nodes()))
        self.is_playing = False
        self.is_thread_alive = False
        self.init_graph()
        self.init_buttons()
        self.init_labels()
        plt.show()


    def init_labels(self):
        if (self.algorithm.get_iteration() != 0):
            text_iteration = "Iteration : "+str(self.algorithm.get_iteration())
        else:
            text_iteration = ""
        self.iterations_label = plt.text(-8,0.2, text_iteration)
        if (self.is_playing):
            self.state_run = plt.text(-5,0.2, "Playing")
        else:
            self.state_run = plt.text(-5,0.2, "Paused")

    def init_graph(self):
        self.ax.clear()
        plt.clf()
        plt.cla()
        for v in range(len(self.algorithm.get_vertices())):
            self.G.add_node(v, pos=self.algorithm.get_vertices()[v])
        for e in self.algorithm.get_edges():
            self.G.add_edge(e[0], e[1], color=e[3], weight=e[2])
        edge_colors = [self.G[u][v]['color'] for u, v in self.G.edges()]
        node_colors = []
        counter = 0
        for node in self.G:
            node_colors.append(self.algorithm.get_neighbors(counter)[1])
            counter += 1

        if (len(self.algorithm.get_vertices()) > 100):
            self.label_edges = False

        self.sizeOfNodes = max(min(500, 550-len(self.algorithm.get_vertices())), 10)
        self.showNodeLabels = True
        if (self.sizeOfNodes < 50):
            self.showNodeLabels = False

        self.width = 3

        if (len(self.algorithm.get_vertices()) > 5000) :
            self.sizeOfNodes = 5
            self.width = 1

        nx.draw(self.G, nx.get_node_attributes(self.G, 'pos'), with_labels=self.showNodeLabels, node_color=node_colors,
                node_size=self.sizeOfNodes, width=self.width, edge_color=edge_colors)
        if self.label_edges:
            edge_labels = dict([((u, v,), d['weight'])
                                for u, v, d in self.G.edges(data=True)])
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=6)
        

    def init_buttons(self):

        def update_graph():
            self.ax.clear()
            plt.clf()
            plt.cla()
            for v in range(len(self.algorithm.get_vertices())):
                self.G.add_node(v, pos=self.algorithm.get_vertices()[v])
            for e in self.algorithm.get_edges():
                self.G.add_edge(e[0], e[1], color=e[3], weight=e[2])
            edge_colors = [self.G[u][v]['color'] for u, v in self.G.edges()]
            node_colors = []
            counter = 0
            for node in self.G:
                node_colors.append(self.algorithm.get_neighbors(counter)[1])
                counter += 1
            nx.draw(self.G, nx.get_node_attributes(self.G, 'pos'), with_labels=self.showNodeLabels, node_color=node_colors,
                    node_size=self.sizeOfNodes, width=self.width, edge_color=edge_colors)
            if self.label_edges:
                edge_labels = dict([((u, v,), d['weight'])
                                    for u, v, d in self.G.edges(data=True)])
                nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=edge_labels, font_size=6)
            self.init_buttons()
            self.init_labels()

        def animate():
            self.algorithm.update()
            update_graph()
            
        def play_thread():
            time.sleep(self.speed/1000)
            while(self.is_playing and self.algorithm.get_remaining_counter() > 0):
                animate()
                plt.draw()
                time.sleep(self.speed/1000)
            self.is_thread_alive = False
            self.is_playing = False
        
        def play(event):
            if (self.is_thread_alive):
                return
            self.is_playing = True
            self.is_thread_alive = True
            thread = threading.Thread(target = play_thread)
            thread.start()

        def pause(event):
            self.is_playing = False
            self.state_run.set_text("Paused")
            plt.show()
        def next(event):
            if (self.is_playing):
                return
            animate()
            plt.show()

        def back(event):
            if (self.is_playing):
                return
            target_counter_history = self.algorithm.get_counter_history() - 1
            self.algorithm.reinitialize_history(copy.deepcopy(self.save_init_config))
            if (target_counter_history <= 1):
                update_graph()
                plt.show()
                return
            while (self.algorithm.update() < target_counter_history):
                pass
            update_graph()
            plt.show()
        def end(event):
            self.is_playing = False
            while(self.algorithm.update() < self.algorithm.get_length_history()):
                pass
            animate()
            plt.show()
        
        self.button_play = widgets.Button(plt.axes([0.7,0,0.1,0.05]), "Play")
        self.button_pause = widgets.Button(plt.axes([0.8,0,0.1,0.05]), "Pause")
        self.button_next = widgets.Button(plt.axes([0.6,0,0.1,0.05]), "Next")
        self.button_back = widgets.Button(plt.axes([0.5,0,0.1,0.05]), "Back")
        self.button_end = widgets.Button(plt.axes([0.9,0,0.1,0.05]), "End")


        self.button_play.on_clicked(play)
        self.button_pause.on_clicked(pause)
        self.button_next.on_clicked(next)
        self.button_back.on_clicked(back)
        self.button_end.on_clicked(end)

