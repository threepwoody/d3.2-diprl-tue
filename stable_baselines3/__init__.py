import os


from stable_baselines3.ppo import PPO


version_file = os.path.join(os.path.dirname(__file__), "version.txt")

with open(version_file) as file_handler:

    __version__ = file_handler.read().strip()


__all__ = ["PPO"]
