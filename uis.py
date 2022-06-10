# # ###Mahar#
# # #Library
# # from datetime import date
# # class Library_System:
# #
# #   data = {235:"Zeeshan",
# #           269:"Awais",
# #           121:"Ali",
# #           123:"Usman"
# #           }
# #
# #   Book_category = {"Biography":["Steve jobs","Elon Musk"],
# #       "Horror":["Haunted","Dawn","The Stand"],
# #        "Fantasy":["The Water Dancer","Ninth House","Circe"],
# #        "Programming":["PF","OOP","Data structures"]
# #       }
# #
# #   def get_book_category(self):
# #     print("All Books category that you select")
# #     print(self.Book_category.keys())
# #     get_category = input("Enter The category of Book: ")
# #     x = self.Book_category.get(get_category)
# #
# #
# #     if x != None:
# #       print("All Books available in this category:")
# #       print(x)
# #       Book_Name = input("Enter The Name of Book: ")
# #       print(f"{Book_Name} is issued to you \nThe Last date is {date.today()}")
# #     else:
# #       print("You Enter Invalid Book Category")
# #
# #   def search_Record(self):
# #     search_name = int(input("Enter Roll Number: "))
# #     x = self.data.get(search_name)
# #
# #     if x != None:
# #       print(f"Hi {x} Welcome to the Library System :")
# #       self.get_book_category()
# #     else:
# #       print("Invalid user")
# #
# # if __name__ == "__main__":
# #   new = Library_System()
# #   new.search_Record()
# #
# # #Graph using List
# # # import all libraries that we needed to implement Graph using list and then plot the Graph
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import networkx as nx
# #
# # # list of edges so call the fuction and pass this edges to make the graph
# # all_edges = [
# #     ("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("C", "E"), ("D", "E")
# # ]
# #
# #
# # # Graph Class where implement all the function related to graph making and priting list also plot the graph
# # class Graph:
# #
# #     # this is constructor where we initilized all values that we are going to use in this class
# #     def __init__(self, nodes):
# #         self.nodes = nodes
# #         self.adj_list = {}
# #
# #         for node in self.nodes:
# #             self.adj_list[node] = []
# #
# #     # this function is used for the add the adges to make graph
# #     def add_edge(self, u, v):
# #         self.adj_list[u].append(v)
# #         self.adj_list[v].append(u)
# #
# #     # print the adjacency list by this fuction
# #     def print_adj_list(self):
# #         for node in self.nodes:
# #             print(node, ":", self.adj_list[node])
# #
# #     # this function is used to plot the graph by using library name newtworkx
# #     def Plot_Graph(self):
# #         G = nx.Graph()
# #         G.add_edges_from(all_edges)
# #         nx.draw(G, with_labels=1)
# #
# #
# # # main function type where our program start actually
# # if __name__ == '__main__':
# #     nodes = ["A", "B", "C", "D", "E"]
# #     graph = Graph(nodes)
# #
# #     # adding adges by calling the function name add_edge() in this loop
# #     for u, v in all_edges:
# #         graph.add_edge(u, v)
# #
# #     # print the adjacency list by calling function that we write in Graph class
# #     print("List\n")
# #     graph.print_adj_list()
# #
# #     # plot the Graph by calling function that we write in Graph Class
# #     print("\nGraph")
# #     graph.Plot_Graph()
# # #Graph using Matrix
# # # Adjacency Matrix representation in Python
# # #import all libraries that we needed to implement Graph using matrix and then plot the Graph
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import networkx as nx
# #
# # #Graph class where implement all the function related to graph making by using matrix and priting matrix also plot the graph
# # class Graph:
# #
# #     # Initialize the matrix of all values with zero(0)
# #     def __init__(self, size):
# #         self.adjMatrix = []
# #         for i in range(size):
# #             self.adjMatrix.append([0 for i in range(size)])
# #         self.size = size
# #
# #     # Add edges
# #     def add_edge(self, v1, v2):
# #         if v1 == v2:
# #             print(f"Same vertex {v1} and {v2}")
# #         self.adjMatrix[v1][v2] = 1
# #         self.adjMatrix[v2][v1] = 1
# #
# #     # Print the matrix
# #     def print_matrix(self):
# #       n=0
# #       print("  0 1 2 3 4")
# #       for row in self.adjMatrix:
# #         print(f"{n} ",end="")
# #         n=n+1
# #         for val in row:
# #           print(f"{val} ",end="")
# #         print("")
# #
# #     # by using this we plot the graph
# #     def Plot_Graph(self):
# #       b=np.matrix(self.adjMatrix)
# #       k=nx.from_numpy_matrix(b)
# #       nx.draw(k,with_labels=1)
# #
# # if __name__ == '__main__':
# #     g = Graph(5)
# #     g.add_edge(0, 1)
# #     g.add_edge(0, 2)
# #     g.add_edge(1, 2)
# #     g.add_edge(2, 0)
# #     g.add_edge(2, 3)
# #     g.add_edge(3, 4)
# #     g.add_edge(4, 0)
# #
# #     print("\nMatrix\n")
# #     g.print_matrix()
# #
# #     print("\nGraph\n")
# #     g.Plot_Graph()
# #
# # #DFS
# # # Python Program to print DFS traversal
# # # from a given source vertex.
# # from collections import defaultdict
# # import networkx as nx
# #
# # # This class represents a directed Tree/graph
# # # using adjacency list representation
# # class Graph:
# #
# #     # Constructor
# #     def __init__(self):
# #
# #         # default dictionary to store Tree/graph
# #         self.graph = defaultdict(list)
# #         self.all_edges=[]
# #
# #         # function to add an edge to Tree/graph
# #     def addEdge(self, u, v):
# #         self.graph[u].append(v)
# #         self.graph[v].append(u)
# #         self.all_edges.append((u,v))
# #
# #         #function to plot the Tree/graph
# #     def Plot_Graph(self):
# #         G=nx.Graph()
# #         G.add_edges_from(self.all_edges)
# #         nx.draw(G,with_labels=1)
# #
# #         # Function to print a DFS of Tree/graph
# #     def DFS(self, s):
# #
# #         # Mark all the vertices as not visited
# #         visited = [False] * (max(self.graph) + 1)
# #
# #         # Create a stack for DFS
# #         stack = [s]
# #
# #         # Mark the source node as
# #         # visited and push/append it in stack
# #         visited[s] = True
# #
# #         while stack:
# #
# #             # pop a vertex from
# #             # stack and print it
# #             s = stack.pop()
# #             print(s, end=" ")
# #
# #             # Get all adjacent vertices of the
# #             # poped vertex s. If a adjacent
# #             # has not been visited, then mark it
# #             # visited and push/append it
# #             for i in self.graph[s]:
# #                 if not visited[i]:
# #                     stack.append(i)
# #                     visited[i] = True
# #
# # # Driver code
# # if __name__ == '__main__':
# #
# #   # Create a graph given in
# #   # the above diagram
# #   g = Graph()
# #   g.addEdge(1, 2)
# #   g.addEdge(1, 3)
# #   g.addEdge(2, 4)
# #   g.addEdge(2, 5)
# #   g.addEdge(3, 6)
# #   g.addEdge(3, 7)
# #
# #   #call function to plot the Tree/Graph
# #   g.Plot_Graph()
# #
# #   #call function to print the DFS of Tree/Graph
# #   print("Following is Depth First Traversal"
# #       " \n(starting from vertex 1)")
# #   g.DFS(1)
# #
# #
# #
# # #DFS with Starting and Ending Node
# # # Python Program to print DFS traversal
# # # from a given source vertex.
# # from collections import defaultdict
# # import networkx as nx
# #
# # # This class represents a directed Tree/graph
# # # using adjacency list representation
# # class Graph:
# #
# #     # Constructor
# #     def __init__(self):
# #
# #         # default dictionary to store Tree/graph
# #         self.graph = defaultdict(list)
# #         self.all_edges=[]
# #
# #         # function to add an edge to Tree/graph
# #     def addEdge(self, u, v):
# #         self.graph[u].append(v)
# #         self.graph[v].append(u)
# #         self.all_edges.append((u,v))
# #
# #         #function to plot the Tree/graph
# #     def Plot_Graph(self):
# #         G=nx.Graph()
# #         G.add_edges_from(self.all_edges)
# #         nx.draw(G,with_labels=1)
# #
# #         # Function to print a DFS of Tree/graph
# #     def DFS(self, s,goal_node):
# #
# #         # Mark all the vertices as not visited
# #         visited = [False] * (max(self.graph) + 1)
# #
# #         # Create a stack for DFS
# #         stack = [s]
# #
# #         # Mark the source node as
# #         # visited and push/append it in stack
# #         visited[s] = True
# #
# #         while stack:
# #
# #             # pop a vertex from
# #             # stack and print it
# #             s = stack.pop()
# #             print(s, end=" ")
# #
# #             if s == goal_node:
# #               break
# #
# #             # Get all adjacent vertices of the
# #             # poped vertex s. If a adjacent
# #             # has not been visited, then mark it
# #             # visited and push/append it
# #             for i in self.graph[s]:
# #                 if not visited[i]:
# #                     stack.append(i)
# #                     visited[i] = True
# #
# # # Driver code
# # if __name__ == '__main__':
# #
# #   # Create a graph given in
# #   # the above diagram
# #   g = Graph()
# #   g.addEdge(1, 2)
# #   g.addEdge(1, 3)
# #   g.addEdge(2, 4)
# #   g.addEdge(2, 5)
# #   g.addEdge(3, 6)
# #   g.addEdge(3, 7)
# #
# #   #call function to plot the Tree/Graph
# #   g.Plot_Graph()
# #
# #   #call function to print the DFS of Tree/Graph
# #   print("Following is Depth Limited Traversal"
# #       " \n(starting from vertex 1)")
# #   g.DFS(1,7)
# #
# # #DLS
# # # Python Program to print DLS traversal
# # # from a given source vertex.
# # from collections import defaultdict
# # import networkx as nx
# #
# # # This class represents a directed Tree/graph
# # # using adjacency list representation
# # class Graph:
# #
# #     # Constructor
# #     def __init__(self):
# #
# #         # default dictionary to store Tree/graph
# #         self.graph = defaultdict(list)
# #         self.all_edges=[]
# #
# #         # function to add an edge to Tree/graph
# #     def addEdge(self, u, v):
# #         self.graph[u].append(v)
# #         self.graph[v].append(u)
# #         self.all_edges.append((u,v))
# #
# #         #function to plot the Tree/graph
# #     def Plot_Graph(self):
# #         G=nx.Graph()
# #         G.add_edges_from(self.all_edges)
# #         nx.draw(G,with_labels=1)
# #
# #         # Function to print a DFS of Tree/graph
# #     def DLS(self, s,goal_node,limit):
# #
# #         # Mark all the vertices as not visited
# #         visited = [False] * (max(self.graph) + 1)
# #
# #         # Create a stack for DFS
# #         stack = [s]
# #
# #         # Mark the source node as
# #         # visited and push/append it in stack
# #         visited[s] = True
# #
# #         while stack:
# #
# #             # pop a vertex from
# #             # stack and print it
# #             s = stack.pop()
# #             print(s, end=" ")
# #             # Get all adjacent vertices of the
# #             # poped vertex s. If a adjacent
# #             # has not been visited, then mark it
# #             # visited and push/append it
# #             for i in self.graph[s]:
# #                 if not visited[i]:
# #                     if s!=goal_node or s<=limit:
# #                         stack.append(i)
# #                         visited[i] = True
# #
# # # Driver code
# # if __name__ == '__main__':
# #   # Create a graph given in
# #   # the above diagram
# #   g = Graph()
# #   g.addEdge(1, 2)
# #   g.addEdge(1, 3)
# #   g.addEdge(2, 4)
# #   g.addEdge(2, 5)
# #   g.addEdge(3, 6)
# #   g.addEdge(3, 7)
# #   #call function to plot the Tree/Graph
# #   g.Plot_Graph()
# #   #call function to print the DFS of Tree/Graph
# #   print("Following is Depth Limited Traversal"
# #       " \n(starting from vertex 1 and 7 is goal node with 3 is limit)")
# #   g.DLS(1,7,3)
# # #UCS
# # import copy
# # import queue as Q
# #
# # class Graph:
# #     def _init_(self):
# #         # graph dictonary contains all the adjacent nodes of each as key and value pair
# #         self.graph = dict()
# #         # cost_dict contains cost of each edge traversal of (u,v)
# #         self.cost_dict = dict()
# #         # final_dict contains all the possible paths from start node s to goal node g with total cost
# #         self.final_dict = dict()
# #         # count_frontier = no. of nodes placed on the frontier
# #         self.count_frontier = 0
# #         # nodes_search = no. of nodes generated during the path search
# #         self.nodes_search = 0
# #         self.visited=[]
# #         self.path_cost=1
# #         self.states_list = set()
# #
# #     # u and v are nodes: edge u-->v & v-->u with cost c (undirectional)
# #     def addEdge(self, u, v, c):
# #         if u not in self.graph:
# #             qu = Q.PriorityQueue()
# #             self.graph.update({u: qu})
# #
# #         self.graph[u].put(v)
# #         self.cost_dict.update({(u, v): c})
# #
# #         if v not in self.graph:
# #             qu = Q.PriorityQueue()
# #             self.graph.update({v: qu})
# #
# #         self.graph[v].put(u)
# #         self.cost_dict.update({(v, u): c})
# #
# #     def UCS_util(self, s, visited, path, goal, value):
# #         # Appending node to the current path
# #         path.append(s)
# #         # Marking that node is visited
# #         visited.append(s)
# #         self.count_frontier+=1
# #         # If goal node is reached save the path and return
# #         if goal == s:
# #             self.final_dict.update({tuple(path): value})
# #             return
# #
# #         # Checking if the adjacent node is been visited and explore the new path if haven't
# #         for i in self.graph[s].queue:
# #             if i not in visited:
# #                 self.states_list.add(i)
# #                 # When new path is being explored add the cost of that path to cost of entire course traversal
# #                 # Send a copy of path list to avoid sending it by reference
# #                 self.UCS_util(i, copy.deepcopy(visited), copy.deepcopy(path), goal, value + self.cost_dict[s, i])
# #
# #     def UCS(self, s, goal):
# #         self.states_list.add(s)
# #         self.visited.append(s)
# #         # List to hold all the nodes visited in path from start node to goal node
# #         path = [s]
# #
# #         for i in self.graph[s].queue:
# #             if i not in self.visited:
# #                 # Make a variable to hold the cost of traversal
# #                 value = self.cost_dict[s, i]
# #                 self.UCS_util(i, copy.deepcopy(self.visited), copy.deepcopy(path), goal, value)
# #
# #     # Display all the paths that is been discovered from start node to Goal node
# #     def all_paths(self):
# #         # Check if there is any path
# #         if bool(self.final_dict):
# #             for i in self.final_dict:
# #                 self.path_cost=self.final_dict[i]
# #             for i in self.final_dict:
# #                 if self.final_dict[i] < self.path_cost:
# #                     self.path_cost=self.final_dict[i]
# #         else:
# #             print("No Path exist between start and goal node")
# #
# #     # Find the most optimal path between start node to goal node
# #     def optimal_path(self):
# #         if bool(self.final_dict):
# #             print("Complete path with states visited from Start to Goal Node: ", min(self.final_dict, key=self.final_dict.get))
# #             print("Total path Cost : ",self.path_cost)
# #         else:
# #             print("No Path exist between start and goal node")
# #
# #
# # g = Graph()
# # g._init_()
# #
# # #Add Edge by this
# # g.addEdge('A','B',3)
# # g.addEdge('A','C',4)
# # g.addEdge('B','I',11)
# # g.addEdge('B','H',10)
# # g.addEdge('B','E',7)
# # g.addEdge('C','D',7)
# # g.addEdge('C','F',9)
# # g.addEdge('D','E',9)
# # g.addEdge('E','F',11)
# # g.addEdge('E','H',13)
# # g.addEdge('F','G',13)
# # g.addEdge('G','H',15)
# # g.addEdge('G','K',18)
# # g.addEdge('H','K',19)
# # g.addEdge('H','L',20)
# # g.addEdge('H','J',18)
# # g.addEdge('I','J',19)
# #
# # #calculate cost from A to J NODE
# # g.UCS('A', 'J')
# # g.all_paths()
# # g.optimal_path()
# # #A*
# #
# # # this function returns heuristic value for all nodes
# # def heuristic(n):
# #     H_dist = {
# #         'A': 40,
# #         'B': 32,
# #         'C': 25,
# #         'D': 35,
# #         'E': 19,
# #         'F': 17,
# #         'G': 0,
# #         'H': 10
# #     }
# #     return H_dist[n]
# #
# #
# # # fuction to return neighbor and its cost from the passed node
# # def get_neighbors(v):
# #     if v in Graph_nodes:
# #         return Graph_nodes[v]
# #     else:
# #         return None
# #
# #
# # # main function for A* Search Algorithm
# # def aStar_Search_Algo(start_node, goal_node):
# #     open_set = set(start_node)
# #     closed_set = set()
# #     # store distance from starting node
# #     g = {}
# #     # parents contains an adjacency map of all nodes
# #     parents = {}
# #
# #     # ditance of starting node from itself is zero
# #     g[start_node] = 0
# #     # start_node is root node i.e it has no parent nodes
# #     # so start_node is set to its own parent node
# #     parents[start_node] = start_node
# #
# #     while len(open_set) > 0:
# #         n = None
# #
# #         # node with lowest f() is found
# #         for v in open_set:
# #             if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
# #                 n = v
# #
# #         if n == goal_node or Graph_nodes[n] == None:
# #             pass
# #         else:
# #             for (m, weight) in get_neighbors(n):
# #                 # nodes 'm' not in first and last set are added to first
# #                 # n is set its parent
# #                 if m not in open_set and m not in closed_set:
# #                     open_set.add(m)
# #                     parents[m] = n
# #                     g[m] = g[n] + weight
# #
# #
# #                 # for each node m,compare its distance from start i.e g(m) to the
# #                 # from start through n node
# #                 else:
# #                     if g[m] > g[n] + weight:
# #                         # update g(m)
# #                         g[m] = g[n] + weight
# #                         # change parent of m to n
# #                         parents[m] = n
# #
# #                         # if m in closed set,remove and add to open
# #                         if m in closed_set:
# #                             closed_set.remove(m)
# #                             open_set.add(m)
# #
# #         if n == None:
# #             print('Path does not exist!')
# #             return None
# #
# #         # if the current node is the goal_node
# #         # then we begin reconstructin the path from it to the start_node
# #         if n == goal_node:
# #             path = []
# #
# #             while parents[n] != n:
# #                 path.append(n)
# #                 n = parents[n]
# #
# #             path.append(start_node)
# #
# #             path.reverse()
# #
# #             print(f'Path found: {path}')
# #             return path
# #
# #         # remove n from the open_list, and add it to closed_list
# #         # because all of his neighbors were inspected
# #         open_set.remove(n)
# #         closed_set.add(n)
# #
# #     print('Path does not exist!')
# #     return None
# #
# #
# # # driven program
# # if __name__ == "__main__":
# #     # make graph here with dictionary
# #     Graph_nodes = {
# #         'A': [('B', 1), ('C', 1), ('D', 7)],
# #         'B': [('E', 1)],
# #         'C': [('E', 8), ('F', 1)],
# #         'D': [('F', 2)],
# #         'E': [('H', 9)],
# #         'F': [('G', 2)],
# #         'H': [('G', 1)],
# #     }
# #     aStar_Search_Algo('A', 'G')
# # #CSP
# # # constraint library included to run build in functions
# # from constraint import *
# #
# # #created the object of given problem
# # problem = Problem()
# #
# # #add value of a by using addVariable build-in function
# # problem.addVariable("a", [1,2,3])
# #
# # #add value of b by using addVariable build-in function
# # problem.addVariable("b", [4,5,3])
# #
# # #get answer by calling getsolution function and save answer in sol
# # sol=problem.getSolutions()
# #
# # #print without contraint answer(this answer we get is without contsraint)
# # print("wihtout constraints",sol)
# #
# # #this answer we get is without contsraint if we
# # #need contraint then we need to add them using below function(addconstraint)
# # problem.addConstraint(lambda a, b: a!= b ,
# #                           ("a", "b"))
# #
# # #Now here we print answer with contraint
# # print("With constraints",problem.getSolutions())
# # #CSP2
# # # constraint library included to run build in functions
# # from constraint import *
# #
# # #created the object of given problem
# # problem = Problem()
# #
# # #add value of a by using addVariable build-in function
# # problem.addVariable("a", [2,4,6,8,10,12])
# #
# # #add value of b by using addVariable build-in function
# # problem.addVariable("b", [3,6,12,15])
# #
# # #get answer by calling getsolution function and save answer in sol
# # sol=problem.getSolutions()
# #
# # #print without contraint answer(this answer we get is without contsraint)
# # print("wihtout constraints",sol)
# #
# # #this answer we get is without contsraint if we
# # #need contraint then we need to add them using below function(addconstraint)
# # problem.addConstraint(lambda a, b: a== b and a==6 ,
# #                           ("a", "b"))
# # #Now here we print answer with contraint
# # print("With constraints",problem.getSolutions())
# # #Alpha beta pruning
# # #Initial values of Aplha and Beta
# # MAX, MIN = 500, -500
# #
# #
# # # Returns optimal value for current player
# # # (Initially called for root and maximizer)
# # def minimax(dep, indexnode, player_maximizing, vals, alpha, beta)
# #     if dep == 3:
# #         return vals[indexnode]
# #     if player_maximizing:
# #         best = MIN
# #
# #     # Recursion for left and right children
# #     for index in range(0, 2):
# #         best = max(best, minimax(dep + 1, indexnode * 2 + index, False, vals, alpha, beta))
# #         alpha = max(alpha, best)
# #                         # Alpha Beta Pruning
# #         if beta <= alpha:
# #           break
# #                     return best
# #                 else:
# #                     best = MAX
# #                     # Recur for left and right children
# #                     for index in range(0, 2):
# #                         best = min(best, minimax(dep + 1, indexnode * 2 + index, True, vals, alpha, beta))
# #                         beta = min(beta, best)
# #                         # Alpha Beta Pruning
# #                         if beta <= alpha:
# #                             break
# #                     return best
# #
# #
# # # Driver Code
# # if __name__ == "__main__":
# #     values = [2, 3, 5, 9, 0, 1, 7, 5]
# #     print("Optimal value is :", minimax(0, 0, True, values, MIN, MAX))
# # #ANN Preceptron Training Rule
# # # import numpy library for array etc.
# # import numpy as np
# #
# # # here we are applying perceptron for OR
# # operator = 'or'
# #
# # # creating input data for training purpose
# # atributes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# #
# # # their respective classes or labels so we train our perceptron by using this
# # labels = np.array([0, 1, 1, 1])
# #
# # # initially we are initialize random values for weights
# # w = [+9, +9]
# #
# # # after calculating value using propagation function then apply activation function by using this threshold
# # threshold = 5
# #
# # # learning rate (in which rate we train our model used in the time of backpropagation for updating weights)
# # alpha = 0.5
# #
# # # learning time (maximum limit for our model training)
# # # if model is not trained in this limit then model stop training
# # epoch = 1000
# #
# # # at initial stage printing learning rate and threshold here
# # print("learning rate: ", alpha, ", threshold: ", threshold)
# #
# # # this loop run from 0 to our maximimum limit means we have 1000 maximum limit here
# # for i in range(0, epoch):
# #     # for every epoch we print here
# #     print("epoch ", i + 1)
# #
# #     # this variable is used to terminate the for loop if learning completed in early epoch
# #     global_delta = 0
# #
# #     # this loop is for the total input dataset
# #     for j in range(len(atributes)):
# #
# #         # actual output for any specific input
# #         actual = labels[j]
# #
# #         # here we are applying propagation function
# #         sum = atributes[j][0] * w[0] + atributes[j][1] * w[1]
# #
# #         # here we are applying threshold which we called activation function in the language of neural network
# #         if sum > threshold:  # then fire
# #             predicted = 1
# #         else:  # do not fire
# #             predicted = 0
# #
# #         # calculating the total difference in our predicted output and our actual output
# #         delta = actual - predicted
# #
# #         # after apply absolute on delta then add with previous golbal_delta for stoping the training of model (if learning completed in early epoch)
# #         global_delta = global_delta + abs(delta)
# #
# #         # update weights with respect to the error
# #         for k in range(0, 2):
# #             w[k] = w[k] + delta * alpha
# #
# #         # printing information about model here
# #         print(atributes[j][0], " ", operator, " ", atributes[j][1], " -> actual: ", actual, ", predicted: ", predicted,
# #               " (w: ", w[0], ")")
# #
# #     # for stop the training of model if learning completed before reaching the maximum epoch limit
# #
# #     if global_delta == 0:
# #         break
# #     print("------------------------------")
# # #ANN with Delta Rule
# # # import numpy library for array etc.
# # import numpy as np
# #
# # # here we are applying perceptron for OR
# # operator = 'or'
# #
# # # creating input data for training purpose
# # atributes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# #
# # # their respective classes or labels so we train our perceptron by using this
# # labels = np.array([0, 1, 1, 1])
# #
# # # initially we are initialize random values for weights
# # w = [+9, +9]
# #
# # # after calculating value using propagation function then apply activation function by using this threshold
# # threshold = 5
# #
# # # learning rate (in which rate we train our model used in the time of backpropagation for updating weights)
# # alpha = 0.5
# #
# # # learning time (maximum limit for our model training)
# # # if model is not trained in this limit then model stop training
# # epoch = 1000
# #
# # # at initial stage printing learning rate and threshold here
# # print("learning rate: ", alpha, ", threshold: ", threshold)
# #
# # # this loop run from 0 to our maximimum limit means we have 1000 maximum limit here
# # for i in range(0, epoch):
# #     # for every epoch we print here
# #     print("epoch ", i + 1)
# #
# #     # this variable is used to terminate the for loop if learning completed in early epoch
# #     global_delta = 0
# #
# #     # this loop is for the total input dataset
# #     for j in range(len(atributes)):
# #
# #         # actual output for any specific input
# #         actual = labels[j]
# #
# #         # here we are applying propagation function
# #         sum = atributes[j][0] * w[0] + atributes[j][1] * w[1]
# #
# #         # in delta rule no comparison
# #         predicted = sum
# #
# #         # calculating the total difference in our predicted output and our actual output
# #         delta = actual - predicted
# #
# #         # after apply absolute on delta then add with previous golbal_delta for stoping the training of model (if learning completed in early epoch)
# #         global_delta = global_delta + abs(delta)
# #
# #         # update weights with respect to the error
# #         for k in range(0, 2):
# #             w[k] = w[k] + delta * alpha
# #
# #         # printing information about model here
# #         print(atributes[j][0], " ", operator, " ", atributes[j][1], " -> actual: ", actual, ", predicted: ", predicted,
# #               " (w: ", w[0], ")")
# #
# #     # for stop the training of model if learning completed before reaching the maximum epoch limit
# #
# #     if global_delta == 0:
# #         break
# #     print("------------------------------")
# # #Min Max
# # class MinMax:
# #     def optimal_result(self,value, turn):
# #         length=len(value)
# #         if(length==1):
# #             return value
# #         elif(turn==1):
# #             newlist=[]
# #             for y in range(0,len(value)-1,1):
# #                 newlist.append(max(value[y],value[y+1]))
# #             value=newlist
# #             turn-=1
# #         else:
# #             newlist=[]
# #             for z in range(0,len(value)-1,1):
# #                 newlist.append(min(value[z],value[z+1]))
# #             value=newlist
# #             turn+=1
# #         print(value)
# #         return M.optimal_result(value,turn)
# #
# # M=MinMax()
# # print(" ********MinMax Algorithm******")
# # #here is the list of values
# # values=[4,6,2,10,14,6,21,22]
# # flag=1
# # # set flag if flag is 1 then it is for max if 0 then it is for 0
# # result=M.optimal_result(values,flag)
# # print("  ===>The optimal Solution in case of Max First is: "+str(result))
# # flag=0
# # result=M.optimal_result(values,flag)
# # print("  ===>The optimal Solution in case of Min First is:  "+str(result))
# # #Hill Climbing
# from collections import defaultdict
# from itertools import permutations
# class Graph:
#     def __init__(self, S, Vert):
#         self.graph = defaultdict(list)
#         self.vertices = Vert
#         self.vertices.remove(S)
#         self.start = S
#     def addEdge(self, u, v):
#         self.graph[u].append(v)
#
#     def printGraph(self):
#         print(self.graph)
#
#     def findactualpathvalue(self):
#         List = []
#         com = permutations(self.vertices)
#         for i in list(com):
#             a = list(i)
#             a.insert(0, self.start)
#             a.append(self.start)
#             List.append(a)
#         return List
#     def generateallsol(self, List):
#         sc = []
#         for i in range(len(List)):
#             S = 0
#             for j in range(len(List[i])-1):
#                 a = List[i][j]
#                 b = List[i][j+1]
#                 c = a+b
#                 S = S + actualvalues[c]
#             sc.append(S)
#         return sc
#
#     def hillClimb(self, list):
#         sh = list[0]
#         for x in range(len(list)):
#             if list[x] < sh:
#                 sh = list[x]
#             else:
#                 break
#         return sh
# # Driver code
# vertices = ["A", "B", "C", "D"]
# g = Graph("A", vertices)
# g.addEdge('A', 'B')
# g.addEdge('A', 'C')
# g.addEdge('A', 'D')
# g.addEdge('B', 'A')
# g.addEdge('B', 'C')
# g.addEdge('B', 'D')
# g.addEdge('C', 'A')
# g.addEdge('C', 'B')
# g.addEdge('C', 'D')
# g.addEdge('D', 'A')
# g.addEdge('D', 'B')
# g.addEdge('D', 'C')
#
# actualvalues = \
#     {'AB': 25,
#      'AD': 15,
#      'BD': 45,
#      'BC': 10,
#      'CD': 5,
#      'AC': 10,
#      'BA': 25,
#      'DA': 15,
#      'DB': 45,
#      'CB': 10,
#      'DC': 5,
#      'CA': 10,
#      }
# print("^^^^^^^^^^^^^Travelling SalesMan Problem^^^^^^^^^^^^^^")
# g.printGraph()
# L = g.findactualpathvalue()
# S = g.generateallsol(L)
# sp = g.hillClimb(S)
# for i in range(len(L)):
#     print(L[i], "  ", S[i])
# print()
# i = S.index(sp)
# print("  ===> The Shortest Path cost is ", sp, " of path ", L[i])
#
