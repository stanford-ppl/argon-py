from argon.types.step import Stop, FVal, FStream
from argon.state import State
from typing import List

def test_stop_token():
    a = Stop(1)
    print(a)
    print(type(a))

def test_fixed_tp_stream():
    state = State()
    with state:
        a = FStream[int]().const([FVal(1.0),FVal(2.0),Stop(1),FVal(3.0),FVal(4.0),Stop(2)])
        print(f"a.C = {a.C}")
        print(f"a.A = {a.A}")
        print(f"a.T = {a.T}")