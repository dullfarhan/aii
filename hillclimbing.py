from collections import defaultdict


class Graph:

    def __init__(self):
        self.allPath = []
        self.V=['A','B','C','D']
        self.graph = defaultdict(list)


    def addEdge(self, u, v):
        self.graph[u].append(v)

    def printGraph(self):
        print(self.graph)

    def findactualpathvalue(self, mypath):
        cost=0
        for i in range(1,len(mypath)):
            edge=mypath[i-1]+mypath[i]
            cost = cost+actualvalues.get(edge)
        return cost


    # find Cost of Current Path

    def generateallsol(self,u,d,visited,path):

        # Mark the current node as visited and store in path
        res_list = [i for i in range(len(self.V)) if self.V[i] == u]

        visited[res_list[0]] = True
        path.append(u)

        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            # print(path)

            self.allPath.append(path.copy())

        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph[u]:
                res_list2 = [j for j in range(len(self.V)) if self.V[j] == i]
                if visited[res_list2[0]] == False:
                   self.generateallsol(i, d, visited, path)

        # Remove current vertex from path[] and mark it as unvisited
        path.pop()

        visited[res_list[0]] = False

    # All paths of TSP

    def hillclimbing(self,s,d):
        # genetrate All posible solutions of paths
        allpaths=[]
        visited = [] ;
        for x in range(0,len(self.V)):
            visited.append(False)

        path = []

        self.generateallsol(s, d, visited, path)

        # Create an array to store paths

# start from a first Path

        for i in range(len(self.allPath)-1):
            fistcost=self.findactualpathvalue(self.allPath[i])
            secondcost=self.findactualpathvalue(self.allPath[i+1])

            if(fistcost<secondcost):
                return self.allPath[i],fistcost

        return self.allPath[-1],secondcost

# Find its Cost of Traveling
# Loop to all other paths and thier cost one by one and Comapre
# if new path better than old than find other path otherwise break


g = Graph()

g.addEdge('A', 'B')
g.addEdge('A', 'C')
g.addEdge('A', 'D')
g.addEdge('B', 'A')
g.addEdge('B', 'C')
g.addEdge('B', 'D')
g.addEdge('C', 'A')
g.addEdge('C', 'B')
g.addEdge('C', 'D')
g.addEdge('D', 'A')
g.addEdge('D', 'B')
g.addEdge('D', 'C')

actualvalues = \
    {'AB': 25,
     'AD': 15,
     'BD': 45,
     'BC': 10,
     'CD': 5,
     'AC': 10,
     'BA': 25,
     'DA': 15,
     'DB': 45,
     'CB': 10,
     'DC': 5,
     'CA': 10,
     }
path,cost =g.hillclimbing('A','D')
print("Hill Climbing")
print("Path",path)
print("cost",cost)
