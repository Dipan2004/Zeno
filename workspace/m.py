from collections import deque

def bfs(graph, start):
    """
    Perform breadth-first search on a graph starting from node 'start'.
    
    :param graph: A dictionary representing the adjacency list of the graph.
                 Keys are nodes and values are lists of adjacent nodes.
    :param start: The starting node for the BFS
    :return: A list of nodes in the order they were visited during BFS, including the start node.
    """
    visited = set()  # Set to keep track of visited nodes
    queue = deque([start])  # Initialize queue with start node
    
    while queue:
        node = queue.popleft()  # Dequeue a node from the queue
        if node not in visited:  # Check if node has already been visited
            visited.add(node)  # Mark the node as visited
            
            # Enqueue all adjacent nodes that have not been visited yet
            queue.extend(graph[node] - visited)
    
    return visited