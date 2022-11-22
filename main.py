import exptree

def f(x):
    return x**3 + 9*x 

from random import random, choice

UNARIES = ["sqrt(%s)", "exp(%s)", "log(%s)", "sin(%s)", "cos(%s)", "tan(%s)",
           "sinh(%s)", "cosh(%s)", "tanh(%s)", "asin(%s)", "acos(%s)",
           "atan(%s)", "-%s"]
BINARIES = ["%s + %s", "%s - %s", "%s * %s", "%s / %s", "%s ** %s"]

PROP_PARANTHESIS = 0.3
PROP_BINARY = 0.7

def generate_expressions(scope, num_exp, num_ops):
    scope = list(scope) # make a copy first, append as we go
    for _ in xrange(num_ops):
        if random() < PROP_BINARY: # decide unary or binary operator
            ex = choice(BINARIES) % (choice(scope), choice(scope))
            if random() < PROP_PARANTHESIS:
                ex = "(%s)" % ex
            scope.append(ex)
        else:
            scope.append(choice(UNARIES) % choice(scope))
    return scope[-num_exp:] # return most recent expressions

if __name__ == '__main__':
    n = int(input('how much samples: '))
    listpar = [f(x) for x in range(n)]
    # print(listpar)
    tree = exptree.Tree()
    tree.parse('x^2 + x +1')
    tree.setVariable('x',2)
    print(tree.evaluate())