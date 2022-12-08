from math import sin, cos
from binarytree import Node
import numpy as np
import numexpr as ne


def operator_map(ch: str, left_sum, right_sum):
    # print(f"{ch=} ({type(ch).__name__}) | {left_sum=} ({type(left_sum).__name__}) | {right_sum=} ({type(right_sum).__name__})" )
    if ch == '+':
        return left_sum + right_sum
    elif ch == '-':
        return left_sum - right_sum
    elif ch == '*':
        return left_sum * right_sum
    elif ch == '/':
        # if right_sum == 0:
        #     return 0
        # TODO : bullshit
        return left_sum / right_sum
    elif ch == '^':
        return left_sum ** right_sum
    elif ch == 'cos':
        return np.cos(right_sum)
    elif ch == 'sin':
        return np.sin(right_sum)
    else:
        pass


def expTree2str(root: Node):
    if root is None:  # empty tree
        return ""
    if root.left is None and root.right is None:  # leaf node
        return str(root.value)
        # TODO : add check for is digit & in vars throw

    left_sum = expTree2str(root.left)
    right_sum = expTree2str(root.right)
    return "(" + left_sum + str(root.value) + right_sum + ")"


def evaluate(root: Node, vars, dbg=0, depth=1):
    # if dbg == 0:
    #     print(root)
    #     print(f"{vars=}")
    # print(f"{root.value=} {depth=}")
    """evaluate value of tree from the input for variables using recursive inorder traverse"""
    if root is None:  # empty tree
        return 0
    if root.left is None and root.right is None:  # leaf node
        if root.value.isnumeric():
            return int(root.value)
            # TODO : add check for is digit & in vars throw
        else:
            return vars

            # TODO : just int not float sin(1.45)
            # TODO : sin is radian
    left_sum = evaluate(root.left, vars, 1, depth+1)
    right_sum = evaluate(root.right, vars, 1, depth+1)
    return operator_map(root.value, left_sum, right_sum)

    # def evaluate_vectorized(root: Node, vars: np.array):
#     if root is None: # empty tree
#         return 0
#     if root.left is None and root.right is None: # leaf node
#         return vars
#         # if root.value in scope:
#         #     return vars[]
#         #     # TODO : add check for is digit & in vars throw
#         # else:
#         #     return int(root.value)
#         #     # TODO : just int not float sin(1.45)
#         #     # TODO : sin is radian
#     left_sum = evaluate_vectorized(root.left, vars)
#     right_sum = evaluate_vectorized(root.right, vars)
#     return operator_map(root.value, left_sum, right_sum)


def evaluate_np(root: Node, vars: np.array, dbg=0, depth=1):
    """evaluate value of tree from the input for variables using recursive inorder traverse"""
    # if dbg == 0:
    #     print(root)
    #     print(f"{vars=}")
    # print(f"{root.value=} {depth=}")

    if root is None:  # empty tree
        return np.zeros(len(vars))
    if root.left is None and root.right is None:  # leaf node
        if root.value.isnumeric():
            return int(root.value)
            # TODO : add check for is digit & in vars throw
        else:
            return vars

            # TODO : just int not float sin(1.45)
            # TODO : sin is radian
    left_sum = evaluate(root.left, vars, 1, depth+1)
    right_sum = evaluate(root.right, vars, 1, depth+1)
    return operator_map(root.value, left_sum, right_sum)
