from diprl.gym_envs.envs.cartpole import (
    ALL_CartPole_FUNCTIONS,
    CartPole_FUNCTIONS,
    CartPole_MODELS,
)


def parse_domain(domain):

    if domain != "CartPole-v1":

        raise ValueError("Use CartPole-v1.")

    return dict(
        models=CartPole_MODELS,
        functions=CartPole_FUNCTIONS,
        all_functions=ALL_CartPole_FUNCTIONS,
        input_dim=4,
        num_action_space=2,
        index_action_space=range(4),
    )
