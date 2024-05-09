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

def test_fixed_tp_stream_zip():
    state = State()
    with state:
        a = Stream().const([Val(1.0),Val(2.0),Stop(1),Val(3.0),Val(4.0),Stop(2)])
        b = Stream().const([Val(1.1),Val(2.1),Stop(1),Val(3.1),Val(4.1),Stop(2)])
        c = a.zip(b)
        print(state)
    

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

