from typing import Tuple


def split(node: Tuple[int, int, int]) -> [
        Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]
]:
    """Splits a node into four quadrants. node is a tuple of (x, y, size)"""
    node_top_left = (node[0], node[1], int(node[2] / 2))
    node_top_right = (int(node[0] + node[2] / 2), node[1], int(node[2] / 2))
    node_bottom_right = (int(node[0] + node[2] / 2), int(node[1] + node[2] / 2),
                                 int(node[2] / 2))
    node_bottom_left = (node[0], int(node[1] + node[2] / 2), int(node[2] / 2))
    return [node_top_left, node_top_right, node_bottom_right, node_bottom_left]

def are_contiguous(node1: Tuple[int, int, int], node2: Tuple[int, int, int]) -> bool:
    if node1[0] <= node2[0]:
        pass
    else:
        node3 = node2
        node2 = node1
        node1 = node3
        pass
    if node1[0] + node1[2] == node2[0]:
        if node1[1]+node1[2] <= node2[1]:
            return False
        elif node1[1] >= node2[1]+node2[2]:
            return False
        else:
            return True
    elif (node1[1] + node1[2] == node2[1]) or (node2[1] + node2[2] == node1[1]):
        if node1[0]+node1[2] <= node2[0]:
            return False
        elif node1[0] >= node2[0]+node2[2]:
            return False
        else:
            return True
    else:
        return False