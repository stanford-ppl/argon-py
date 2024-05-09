from argon.types.token import Stop, Val, Token, Stream
from argon.state import State


def test_stop_token():
    a = Stop(1)
    print(a)
    print(type(a))

def test_token():
    a = Token(Stop(1))
    b = Token(Val(1.0))
    print(a)
    print(type(a))
    print(b)
    print(type(b))

def test_fixed_tp_stream():
    state = State()
    with state:
        a = Stream().const([Val(1.0),Val(2.0),Stop(1),Val(3.0),Val(4.0),Stop(2)])
        print(a)
        print(f"a.C = {a.C}")
        print(f"a.A = {a.A}")

