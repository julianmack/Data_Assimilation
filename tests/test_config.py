from VarDACAE.settings import base as config
import os
import pytest
import numpy as np


class TestConfigEnvVariables():
    """Check environmental variables are properly set"""
    def test_env_set(self):
        settings = config.Config() #should initialize env vars
        seed = os.environ.get("SEED")

        assert seed is not None

    def test_env_set_subclass(self):
        settings = config.ConfigExample() #should initialize env vars
        seed = os.environ.get("SEED")
        assert seed is not None

    def test_env_override(self):
        settings1 = config.Config() #should initialize env vars
        seed1 = int(os.environ.get("SEED"))
        seed_1 = settings1.SEED

        settings1.SEED = 100
        settings1.export_env_vars()
        seed2 = int(os.environ.get("SEED"))

        assert seed1 == seed_1
        assert seed1 != seed2