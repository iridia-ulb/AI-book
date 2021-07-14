from FileHandler import FileHandler
from hashlib import new
import random

class InstancesGenerator:
    """
    This class allows to generate instances with given parameters
    """

    def __init__(self, numberOfNodes, minX = 0, minY = 0, maxX = 50, maxY = 50, probabilityOfStrangeMove = 0.05, minConnexions = 1, maxConnexions = 5, minRadius = 5, maxRadius = 20):
        self.numberOfNodes = numberOfNodes
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
        self.probabilityOfStrangeMove = probabilityOfStrangeMove
        self.minConnexions = minConnexions
        self.maxConnexions = maxConnexions
        self.minRadius = minRadius
        self.maxRadius = maxRadius


    def getEuclideanDistance(self, a, b):
        """
        Gives the euclidean distance between two nodes
        """
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(1/2)

    def getClosestNodes(self, node, vertices):
        """
        Give the list of nodes sorted by distance
        """
        nodes = []
        for i in vertices:
            v = vertices[i]
            if (v == node):
                continue
            nodes.append((i, self.getEuclideanDistance(v, node)))

        sortedNodes = sorted(nodes, key = lambda k: k[1])
        return sortedNodes


    def linkNodes(self,edges, vertices):
        """
        Create the edges between the nodes
        """
        markedNodes = []
        openNodes = []

        for i in vertices:
            openNodes.append(i)
        markedNodes.append(0)
        openNodes.remove(0)

        while( len(openNodes) != 0):
            tempList=[]
            newNodeMark = False
            for i in markedNodes: #Search all the new nodes to mark
                if(i not in tempList):
                    tempList.append(i)
                for e in edges:
                    if (e[0] == i):
                        if(e[1] not in tempList):
                            tempList.append(e[1])
                    elif (e[1] == i):
                        if(e[0] not in tempList):
                            tempList.append(e[0])

            for element in tempList: #Add temp element marked to the definitive marked list
                if(element in openNodes):
                    markedNodes.append(element)
                    openNodes.remove(element)
                    newNodeMark = True
            if(not newNodeMark and len(openNodes) !=0):
                newNode = openNodes.pop()
                closestNodes = self.getClosestNodes(vertices[newNode],vertices)
                for i in range(0,len(closestNodes)):
                    if(closestNodes[i][0] in markedNodes):
                        edges.add((newNode,closestNodes[i][0])) #Add a link between the picked node and the closest node
                        break
                markedNodes.append(newNode)
            tempList.clear()



    def generate(self, file, n_recursion = 0):
        """
        Generate the instance
        """
        V = self.numberOfNodes
        vertices = {}

        posX = random.randint(self.minX, int((self.maxX-self.minX)/3) + self.minX)
        posY = random.randint(self.minY, int((self.maxY-self.minY)/3) + self.minY)
        vertices[0] = (posX, posY)

        posX = random.randint(int(2*(self.maxX-self.minX)/3) + self.minX, self.maxX)
        posY = random.randint(int(2*(self.maxY-self.minY)/3) + self.minY, self.maxY)
        vertices[1] = (posX, posY)

        for i in range(2, V):
            posX = random.randint(self.minX, self.maxX)
            posY = random.randint(self.minY, self.maxY)
            isValid = True
            for j in vertices:
                v = vertices[j]
                if (self.getEuclidianDistance(v, (posX, posY)) < self.minRadius):
                    if (random.random() > self.probabilityOfStrangeMove):
                        isValid = False
                        break
            if (isValid):
                vertices[i] = (posX, posY)
                

        edges = set()
        encounteredVertices = set()
        verticesCounter = {}

        for k in vertices:
            verticesCounter[k] = 0

        for k in vertices:
            v = vertices[k]
            numberOfConnexions = random.randint(self.minConnexions, self.maxConnexions)
            closestNodes = self.getClosestNodes(v, vertices)
            for i in range(numberOfConnexions):
                if (verticesCounter[k] >= self.maxConnexions):
                    break
                currentNodeDistance = closestNodes[i]
                currentNode = currentNodeDistance[0]
                if ((currentNode, k) in edges or (k, currentNode) in edges):
                    continue
                currentDistance = currentNodeDistance[1]
                if (currentDistance > self.maxRadius):
                    if (random.random() < self.probabilityOfStrangeMove):
                        edges.add((k, currentNode))
                        encounteredVertices.add(k)
                        encounteredVertices.add(currentNode)
                        verticesCounter[k] += 1
                        verticesCounter[currentNode] += 1
                else:
                    edges.add((k, currentNode))
                    encounteredVertices.add(k)
                    encounteredVertices.add(currentNode)
                    verticesCounter[k] += 1
                    verticesCounter[currentNode] += 1

        newVertices = {}
        for i in vertices:
            if (i in encounteredVertices):
                newVertices[i] = vertices[i]
        
        vertices = newVertices
        if (0 in newVertices and 1 in newVertices):
            V = len(newVertices)
            vertices = {}
            keys = list(encounteredVertices)
            for i in range(len(encounteredVertices)):
                vertices[i] = newVertices[keys[i]]
                edgesToRemove = []
                for e in edges:
                    if (e[0] == keys[i]):
                        edgesToRemove.append((e, (i, e[1])))
                    elif (e[1] == keys[i]):
                        edgesToRemove.append((e, (e[0], i)))
                for e in edgesToRemove:
                    old = e[0]
                    new = e[1]
                    edges.remove(old)
                    edges.add(new)

            self.linkNodes(edges,vertices)
            fileHandle = FileHandler(file_name=file)
            fileHandle.write(V, vertices, edges)
        
        else:
            n_recursion += 1
            if (n_recursion < 10):
                print("Error while generating the file, retry")
                self.generate(file, n_recursion)
            else:
                print("Too many errors, STOP")
        
        


        