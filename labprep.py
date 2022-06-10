class Graph:
    def __init__(self):
        self.graph = {}
        self.NoOfVertices = 0

    def isComplete(self):
        for n in self.graph:
            if self.graph.get(n) != self.graph.keys():
                print("incomplete Graph")
                return 0
        print("Complete Graph")
        return 0



    def BFS(self, start, goal):
        visited = []
        queue = []
        bfs = []
        visited.append(start)
        queue.append(start)

        while len(queue) != 0 and goal not in bfs:
            first_out = queue.pop(0)
            bfs.append(first_out)

            for neighbour in self.graph[first_out]:

                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
        return bfs

    def DFSUtil(self, v, visited):

        # Mark the current node as visited
        # and print it
        visited.add(v)
        print(v, end=' ')

        # Recur for all the vertices
        # adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, v):

        # Create a set to store visited vertices
        visited = set()

        # Call the recursive helper function
        # to print DFS traversal
        self.DFSUtil(v, visited)

    def add_vertex(self, newVertex):

        if newVertex in self.graph:
            print("Vertex ", newVertex, " already exists.")
        else:
            self.NoOfVertices = self.NoOfVertices + 1
            self.graph[newVertex] = []

    def add_edge(self, vertex1, vertex2):

        if vertex1 not in self.graph:
            print("Vertex1 ", vertex1, " not exist.")

        elif vertex2 not in self.graph:
            print("Vertex2", vertex2, "not exist.")

        else:
            if vertex2 not in self.graph[vertex1]:
                self.graph[vertex1].append(vertex2)
            if vertex1 not in self.graph[vertex2]:
                self.graph[vertex2].append(vertex1)

    def display(self):
        print("\033[91m {}\033[00m".format("  Graph"))
        print("\033[91m {}\033[00m".format(self.graph))


g1 = Graph()

g1.add_vertex(1)
g1.add_vertex(2)
g1.add_vertex(3)
g1.add_vertex(4)
g1.add_vertex(5)

g1.add_edge(1, 3)
g1.add_edge(3, 2)
g1.add_edge(4, 3)
g1.add_edge(10, 5)
g1.add_edge(5, 2)
g1.add_edge(5, 3)
g1.display()
print("BFS: ", g1.BFS(5, 4))
print("DFS: ")
g1.DFS(5)