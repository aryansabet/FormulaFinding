from random import random, choice, randint
from pprint import pprint
from math import sin, cos
from expression_tree import Expression
from binarytree import Node
import numpy as np


def f(x):
    return x**x + x


op = {'+', '-', '*', '/', '^',
      'sin', 'cos'}
op_info = {'+': (2, 1), '-': (2, 1),
           '*': (2, 2), '/': (2, 2),
           '^': (2, 3),
           'sin': (1, 4), 'cos': (1, 4)}
assotiation = {'+': 'LR', '-': 'LR',
               '*': 'LR', '/': 'LR',
               '^': 'RL',
               'sin': 'RL', 'cos': 'RL'}
varchar = {'x'}
strscope = 'x'
scope = [c for c in strscope]


UNARIES = ["sqrt(%s)", "exp(%s)", "log(%s)", "sin(%s)", "cos(%s)", "tan(%s)",
           "sinh(%s)", "cosh(%s)", "tanh(%s)", "asin(%s)", "acos(%s)",
           "atan(%s)", "-%s"]
BINARIES = ["%s + %s", "%s - %s", "%s * %s", "%s / %s", "%s ^ %s"]

PROP_PARANTHESIS = 0.4
PROP_BINARY = 1.0


def generate_expressions(scope, num_exp, num_ops):
    print(f"{num_ops=}")
    print(f"{num_exp=}")

    scope = list(scope)  # make a copy first, append as we go
    for _ in range(num_ops):
        if random() < PROP_BINARY:  # decide unary or binary operator
            ex = choice(BINARIES) % (choice(scope), choice(scope))
            if random() < PROP_PARANTHESIS:
                ex = "(%s)" % ex
            scope.append(ex)
        else:
            scope.append(choice(UNARIES) % choice(scope))
    return scope[-num_exp:]  # return most recent expressions


def operator_map(ch: str, left_sum, right_sum):
    if ch == '+':
        return left_sum + right_sum
    elif ch == '-':
        return left_sum - right_sum
    elif ch == '*':
        return left_sum * right_sum
    elif ch == '/':
        # if right_sum == 0:
        #     return 0
        #todo : bullshit
        return left_sum / right_sum
    elif ch == '^':
        return left_sum ** right_sum
    elif ch == 'cos':
        return cos(right_sum)
    elif ch == 'sin':
        return sin(right_sum)
    else:
        pass


def evaluate(root: Node, vars: dict):
    """evaluate value of tree from the input for variables using recursive
    inorder traverse
    """
    # empty tree
    if root is None:
        return 0
    # leaf node
    if root.left is None and root.right is None:
        if root.value in vars:
            return vars[root.value]
            # TODO : add check for is digit & in vars throw
        else:
            return int(root.value)
            # TODO : just int not float sin(1.45)
            # TODO : sin is radian

    left_sum = evaluate(root.left, vars)
    right_sum = evaluate(root.right, vars)
    return operator_map(root.value, left_sum, right_sum)


def MSE(actual, predictions):
    """Mean Squared Error"""
    return np.square(np.subtract(actual, predictions)).mean()


def roulette_wheel_selection(population, fit_vec: np.array):
    """Roulette Wheel Selection for minimizing problem"""
    # todo : negetive mse probability
    probmax = np.sum(fit_vec)
    selection_probs = 1 -  fit_vec / probmax
    return population[np.random.choice(len(population), p=selection_probs)]


# def mutate(child):
#     pass


def crossover(x: Node, y):
    print("______________________________________________________")
    print(x)
    print(y)
    inorder_x = x.inorder
    inorder_y = y.inorder
    print(inorder_x, '\n',inorder_y)
    x_node = Node(1)
    y_node = Node(1)
    print(f"{x_node=}")
    print(f"{y_node=}")

    while x_node.value not in op:
        x_node = choice(inorder_x)
    while y_node.value not in op:
        y_node = choice(inorder_y)
    print(x_node)
    print(y_node)
    temp =Node(1)
    temp.right = x_node.right
    temp.left = x_node.left
    temp.value = x_node.value
    x_node.left = y_node.left
    x_node.right = y_node.right
    x_node.value = y_node.value
    y_node.left = temp.left
    y_node.right = temp.right
    y_node.value = temp.value
    print(x)
    print(y)   


def genetic_algo(population, mse, mutation_probability):
    i = 0
    while i < 5:
        new_population = []
        for i in range(len(population)):
            x = roulette_wheel_selection(population, mse)
            y = roulette_wheel_selection(population, mse)
            child = crossover(x, y)
            # if (random() < mutation_probability):
            #     mutate(child)
            new_population.append(child)
        i += 1


def main():
    # test()
    # np.set_printoptions(precision=3)

    x = np.array(range(1, 10), dtype='float')
    y = f(x)
    print(f"{x=}")
    print(f"{y=}")
    print(f"{scope=}")
    randexpr_arr = generate_expressions(scope, num_exp=5, num_ops=5)
    randexpr_exp = []
    randexpr_tree = []
    y_pred = []

    for index, expression in enumerate(randexpr_arr):
        test = Expression(expression=expression, operators=op, operators_info=op_info,
                          operators_associativity=assotiation, variables=varchar)
        randexpr_exp.append(test)
        randexpr_tree.append(test.tree())
        y_pred.append(evaluate_vectorized(randexpr_tree[index], x))
        print(f"{index}:", randexpr_exp[index], "& ", expression)
        pprint(y_pred[index])
        print(randexpr_tree[index])

    mse = np.array([MSE(y, gg) for gg in y_pred])
    print(mse)

    genetic_algo(population=randexpr_tree, mse=mse, mutation_probability=0)
# 3: (x+(x^x-x))-(x^x-x) &  (x + (x ^ x - x)) - (x ^ x - x)
# array([1., 2., 3., 4., 5., 6., 7., 8., 9.])

#     __________-______
#    /                 \
#   +______           __-
#  /       \         /   \
# x       __-       ^     x
#        /   \     / \
#       ^     x   x   x
#      / \
#     x   x


def evaluate_vectorized(root: Node, vars: np.array):

    # empty tree
    if root is None:
        return 0
    # leaf node
    if root.left is None and root.right is None:
        return vars
        # if root.value in vars:
        #     return vars[root.value]
        #     # TODO : add check for is digit & in vars throw
        # else:
        #     return int(root.value)
    # TODO : just int not float sin(1.45)
    # TODO : sin is radian

    # evaluate left tree
    left_sum = evaluate_vectorized(root.left, vars)
    # evaluate right tree
    right_sum = evaluate_vectorized(root.right, vars)
    # check which operation to apply
    return operator_map(root.value, left_sum, right_sum)


def test():
    # strscope = "abcde"
    # scope = [c for c in strscope]
    # num_exp=randint(1,5),
    # randexpr = generate_expressions(scope,num_exp=int(input("enter num_exp: ")),num_ops=int(input("enter num_ops: ")))
    # for index,expression in enumerate(randexpr):
    #     print(expression)
    #     # print(f'{index=}------>',tree)

    # test = Expression(expression = "x+sin(90)^2*y",
    #                     operators = {'+', 'sin', '^', '*'},
    #                     operators_info = {'+': (2, 1), '*': (2, 2),'^': (2, 3), 'sin': (1, 4)},
    #                     operators_associativity = {'+': 'LR', '*': 'LR','^': 'RL', 'sin': 'RL'},
    #                     variables = {'x', 'y'})
    expr = "x+sin(90)^2*y"
    expr = '1+2*x^9/cos(x^2)+2*y^9'
    expr = '1+2*x^9/cos(x^2)^2+2*y^9'
    expr = "x+2*y^2"

    op = {'+', '-', '*', '/', '^',
          'sin', 'cos'}
    op_info = {'+': (2, 1), '-': (2, 1),
               '*': (2, 2), '/': (2, 2),
               '^': (2, 3),
               'sin': (1, 4), 'cos': (1, 4)}
    assotiation = {'+': 'LR', '-': 'LR',
                   '*': 'LR', '/': 'LR',
                   '^': 'RL',
                   'sin': 'RL', 'cos': 'RL'}

    varchar = {'x', 'y'}
    test = Expression(expression=expr, operators=op, operators_info=op_info,
                      operators_associativity=assotiation, variables=varchar)
    print(expr)
    print(test.tree())
    # pprint(test._tokens)
    # pprint(list(expr))
    print(evaluate(test.tree(), {'x': 4, 'y': 2}))


if __name__ == '__main__':
    main()
