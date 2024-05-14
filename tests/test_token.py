from argon.types.token import Stop, FVal, FStream
from argon.state import State


def test_stop_token():
    a = Stop(1)
    print(a)
    print(type(a))

def test_fixed_tp_stream():
    state = State()
    with state:
        a = FStream[int]().const([FVal(1.0),FVal(2.0),Stop(1),FVal(3.0),FVal(4.0),Stop(2)])
        print(a)
        print(f"a.C = {a.C}")
        print(f"a.A = {a.A}")
        print(f"a.T = {a.T}")

# def test_fixed_tp_stream_zip():
#     state = State()
#     with state:
#         a = fStream().const([FVal(1.0),FVal(2.0),Stop(1),FVal(3.0),FVal(4.0),Stop(2)])
#         b = fStream().const([FVal(1.1),FVal(2.1),Stop(1),FVal(3.1),FVal(4.1),Stop(2)])
#         c = a.zip(b)
#         print(state)
    

    ''' 
    # Problem1: All the streams are not necessarily the same type of stream 
                (tp should not just be Stream. It should be something like Stream<float,2> or Stream<Buff<float,2>,3>)
    # Problem2: The datatype & rank should be part of the stream
        - datatype: generic
        - rank: constant generics
    State( 
    |   scope=Scope( 
    |   |   parent=None, 
    |   |   symbols=[
    |   |   |   x0 = Zip(Const([Val(value=1.0), Val(value=2.0), Stop(level=1), Val(value=3.0), Val(value=4.0), Stop(level=2)]), Const([Val(value=1.1), Val(value=2.1), Stop(level=1), Val(value=3.1), Val(value=4.1), Stop(level=2)]))( 
    |   |   |   |   tp: Stream  -> This should be something like Stream<float,2> or Stream<Buff<float,2>,3>
    |   |   |   |   ctx: /home/ginasohn/Desktop/research/argon-py/tests/test_token.py:31:12 
    |   |   |   )
    |   |   ], 
    |   |   cache={} 
    |   ), 
    |   prev_state=None 
    )
    '''

