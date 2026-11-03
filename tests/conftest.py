"""Test configuration."""

import pytest


@pytest.fixture(scope="session")
def loop_bin():
    """Define `loop` bin."""
    return "tommos loop"


@pytest.fixture(scope="session")
def mesh_bin():
    """Define `mesh` bin."""
    return "tommos mesh"
