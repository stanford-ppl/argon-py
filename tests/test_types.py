
from argon.types import IntType


def test_inttype():
    assert IntType.L() is int
    assert IntType.R() is IntType

