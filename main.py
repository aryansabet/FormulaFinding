from random import random, choice, randint
from pprint import pprint
from math import sin, cos
from expression_tree import Expression, Node


def f(x):
    return x**3 + 9*x + sin(2*x)


# UNARIES = ["sqrt(%s)", "exp(%s)", "log(%s)", "sin(%s)", "cos(%s)", "tan(%s)",
#            "sinh(%s)", "cosh(%s)", "tanh(%s)", "asin(%s)", "acos(%s)",
#            "atan(%s)", "-%s"]
# BINARIES = ["%s + %s", "%s - %s", "%s * %s", "%s / %s", "%s ^ %s"]

# PROP_PARANTHESIS = 0.3
# PROP_BINARY = 1.0

# # def generate_expressions2(scope, num_exp, num_ops):
# #     scope = list(scope) # make a copy first, append as we go
# #     for _ in range(num_ops):
# #         if random() < PROP_BINARY: # decide unary or binary operator
# #             ex = choice(BINARIES) % (choice(scope), choice(scope))
# #             if random() < PROP_PARANTHESIS:
# #                 ex = "(%s)" % ex
# #             scope.append(ex)
# #         else:
# #             scope.append(choice(UNARIES) % choice(scope))
# #     return scope[-num_exp:] # return most recent expressions


# def generate_expressions(scope, num_exp, num_ops):
#     print(f"{num_ops=}")
#     print(f"{num_exp=}")

#     scope = list(scope)  # make a copy first, append as we go
#     for _ in range(num_ops):
#         if random() < PROP_BINARY:  # decide unary or binary operator
#             ex = choice(BINARIES) % (choice(scope), choice(scope))
#             if random() < PROP_PARANTHESIS:
#                 ex = "(%s)" % ex
#             scope.append(ex)
#         else:
#             scope.append(choice(UNARIES) % choice(scope))
#     return scope[-num_exp:]  # return most recent expressions

def operator_map(ch: str, left_sum, right_sum):
    if ch == '+':
        return left_sum + right_sum
    elif ch == '-':
        return left_sum - right_sum
    elif ch == '*':
        return left_sum * right_sum
    elif ch == '/':
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

    # evaluate left tree
    left_sum = evaluate(root.left, vars)
    # evaluate right tree
    right_sum = evaluate(root.right, vars)
    # check which operation to apply
    return operator_map(root.value, left_sum, right_sum)


def main():

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
