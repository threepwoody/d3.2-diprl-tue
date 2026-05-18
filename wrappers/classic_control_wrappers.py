import gymnasium as gym


class StateClassicControlEnv(gym.Wrapper):

    def __init__(self, env: gym.Env):

        env_name = (
            env.unwrapped.spec.id if getattr(env.unwrapped, "spec", None) else None
        )

        if env_name != "CartPole-v1":

            raise ValueError("Use CartPole-v1.")

        super().__init__(env)
