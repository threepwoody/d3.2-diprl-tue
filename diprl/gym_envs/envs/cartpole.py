def f0_cartpole(x):

    return x[:, 0:1]


def f1_cartpole(x):

    return x[:, 1:2]


def f2_cartpole(x):

    return x[:, 2:3]


def f3_cartpole(x):

    return x[:, 3:4]


CartPole_FUNCTIONS = [
    (f0_cartpole, 1),
    (f1_cartpole, 1),
    (f2_cartpole, 1),
    (f3_cartpole, 1),
]


ALL_CartPole_FUNCTIONS = [([1, 1, 1, 1], 4)]


CartPole_MODELS = []

for name in ["left", "right"]:

    CartPole_MODELS.append(name)
