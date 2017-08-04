leetcode DFS/BFS框架
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        if not node:
            return
        nodeCopy = UndirectedGraphNode(node.label)
        dic = {node: nodeCopy}
        queue = [node]
        # self.bfs(queue, dic)
        self.dfs(queue, dic)
        return nodeCopy
        
    def bfs(self, queue, dic):
        while queue:
            front = queue.pop()
            for neighbor in front.neighbors:
                if neighbor not in dic: 
                    # neighor not visited
                    neighborCopy = UndirectedGraphNode(neighbor.label)
                    dic[neighbor] = neighborCopy
                    dic[front].neighbors.append(neighborCopy)
                    queue.insert(0, neighbor)
                else:
                    # neighbor visited
                    dic[front].neighbors.append(dic[neighbor])
                    
    def dfs(self, stack, dic):
        while stack:
            front = stack.pop()
            for neighbor in front.neighbors:
                if neighbor not in dic:
                    neighborCopy = UndirectedGraphNode(neighbor.label)
                    dic[neighbor] = neighborCopy
                    dic[front].neighbors.append(neighborCopy)
                    stack.append(neighbor)
                else:
                    dic[front].neighbors.append(dic[neighbor])

