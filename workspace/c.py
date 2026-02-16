from collections import deque

def bfs(graph, start):
    """
    Perform breadth-first search on a graph starting from the given node.
    
    :param graph: A dictionary representing the adjacency list of the graph where keys are nodes and values are lists 
                  of neighboring nodes.
    :param start: The starting node for the search.
    :return: A list of nodes visited during the breadth-first search, in the order they were visited.
    """
    visited = set()  # Set to keep track of visited nodes
    queue = deque([start])  # Initialize queue with the start node
    
    while queue:
        node = queue.popleft()  # Dequeue a node from the queue
        
        if node not in visited:
            visited.add(node)  # Mark the node as visited
            
            # Enqueue all adjacent nodes that have not been visited
            queue.extend(graph[node] - visited)
    
    return visited