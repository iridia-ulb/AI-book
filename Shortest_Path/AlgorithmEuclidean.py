from Algorithm import Algorithm


class AlgorithmEuclidean(Algorithm):

    def heuristic(self, a, b):
        node_a = self.get_vertex(a)
        node_b = self.get_vertex(b)
        
        return ((node_a[0] - node_b[0])**2 + (node_a[1] - node_b[1])**2)**(1/2)

