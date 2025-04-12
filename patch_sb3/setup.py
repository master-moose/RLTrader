from setuptools import setup, find_packages

setup(
    name="patch_sb3",
    version="0.1",
    packages=find_packages(),
    description="Patched version of stable-baselines3 PPO to fix tensor.item() warnings",
) 