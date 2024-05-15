from argon.types.step import Stop, FVal, FStream, UStream
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
        print(f"a.A.T = {a.A().T}")
    print(state)

def test_fixed_tp_stream2():
    state = State()
    with state:
        breakpoint()
        a = UStream[str]().const([FVal(1.0),FVal(2.0),Stop(1),FVal(3.0),FVal(4.0),Stop(2)])
        #print(f"a.C = {a.C}")
        #print(f"a.A = {a.A}")
        print("we will start")
        print(f"a.U = {a.U}")
        #print(f"a.A().U = {a.A().U}")
        # print(f"a={a}")
        # print(f"a.A()={a.A()}")
    print(state)