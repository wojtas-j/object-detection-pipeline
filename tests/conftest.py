import os

def pytest_configure(config):
    """ Set PYTEST_RUNNING environment variable during tests. """
    os.environ["PYTEST_RUNNING"] = "1"

def pytest_unconfigure(config):
    """ Clean up PYTEST_RUNNING environment variable after tests. """
    os.environ.pop("PYTEST_RUNNING", None)
