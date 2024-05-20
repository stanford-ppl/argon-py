from argon.types.step import Stop, FVal, FStream, UStream, GStream
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
        print(f"a.C = {a.C}") # List[Union[FVal,Stop]
        print(f"a.A = {a.A}") # FStream[int]
        print(f"a.T = {a.T}") # int
        print(f"a.A.T = {a.A().T}") # int
    print(state)

def test_debug_generic1():
    state = State()
    with state:
        a = UStream[str]()
        print(a)
        print(f"type(a)={type(a)}")                     # <class 'argon.types.step.UStream'>
        print(f"a.__orig_class__={a.__orig_class__}")   # argon.types.step.UStream[str]
        print(f"a.C = {a.C}")   # a.C = typing.List[typing.Union[argon.types.step.FVal, argon.types.step.Stop]]
        print(f"a.A = {a.A}")   # a.A = argon.types.step.UStream[int]
        print(f"a.U = {a.U}")   # a.U = <class 'str'>
    print(state)

def test_debug_generic2():
    state = State()
    with state:
        a = UStream[str]().const([FVal(1.0),FVal(2.0),Stop(1),FVal(3.0),FVal(4.0),Stop(2)])
        # This becomes type A (i.e., UStream[int] because const returns type A)

        print(a)                                        # Const([FVal(value=1.0), FVal(value=2.0), Stop(level=1), FVal(value=3.0), FVal(value=4.0), Stop(level=2)])
        print(f"type(a)={type(a)}")                     # <class 'argon.types.step.UStream'>
        print(f"a.C = {a.C}")           # typing.List[typing.Union[argon.types.step.FVal, argon.types.step.Stop]]
        print(f"a.A = {a.A}")           # argon.types.step.UStream[int]
        print(f"a.A().U = {a.A().U}")   # <class 'int'>  -- a.A() is equivalent to UStream[int]()
        print(f"a.U = {a.U}")           # <class 'int'>
    print(state)


def test_gstream():
    state = State()
    with state:
        a = GStream[int]().const([1,2,3])
        print(f"type(a)={type(a)}")
        print(f"a.C = {a.C}")
        print(f"a.A = {a.A}")
        print(f"a.TP = {a.TP}")
        print(f"a.A().TP = {a.A().TP}")
    print(state)
