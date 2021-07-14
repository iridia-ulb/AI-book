from Algorithm import Algorithm


class AlgorithmMean(Algorithm):


    def heuristic(self, a, b):
        node_a = self.get_vertex(a)
        node_b = self.get_vertex(b)
        dx = abs(node_a[0] - node_b[0])
        dy = abs(node_a[1] - node_b[1])

        return (dx + dy)/2

