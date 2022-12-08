from random import random, choice

UNARIES = ["sqrt(%s)", "exp(%s)", "log(%s)", "sin(%s)", "cos(%s)", "tan(%s)",
           "sinh(%s)", "cosh(%s)", "tanh(%s)", "asin(%s)", "acos(%s)",
           "atan(%s)", "-%s"]
BINARIES = ["%s + %s", "%s - %s", "%s * %s", "%s / %s", "%s ^ %s"]
BINARIES = ["%s + %s", "%s - %s", "%s * %s"]


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