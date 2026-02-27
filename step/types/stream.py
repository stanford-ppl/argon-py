from pydantic.dataclasses import dataclass
from typing import Union, Tuple, override, List, TypeVar, Callable, Any
from argon.ref import Ref
from argon.state import stage
from argon.srcctx import SrcCtx
from .rankgen import RankGen

@dataclass
class Stop:
    level: int

    def __str__(self) -> str:
        return f"S{self.level}"

VT = TypeVar("VT")

@dataclass
class Val[VT]:
    value: VT

    def __str__(self) -> str:
        return str(self.value)
    
@dataclass
class Index:
    value: int

    def __str__(self) -> str:
        return str(self.value)


ST = TypeVar("ST")
ST2 = TypeVar("ST2")
SRK = TypeVar("SRK")
SRK2 = TypeVar("SRK2")


class RStream[ST,ST2,SRK,SRK2](Ref[List[Union[Val[ST], Stop]], "RStream[ST,ST2,SRK,SRK2]"]):
    @override
    def fresh(self) -> "RStream[ST,ST2,SRK,SRK2]":
        return RStream[self.ST, self.ST2, self.SRK, self.SRK2]()
    
    def accum(self, func: Callable[[ST,ST2], SRK]) -> "Stream[ST2,SRK-SRK2]":
        from ..ops.accum import Accum
        
        d = int(self.SRK.__name__[1:]) - int(self.SRK2.__name__[1:])
        assert d >= 0
        return stage(Accum[self.ST, self.ST2, self.SRK, self.SRK2, RankGen().get_rank(d)](self, func), ctx=SrcCtx.new(2))
    
    def repeat(self, other: "RStream[ST,ST2,SRK,SRK2]") -> "Stream[ST,SRK+1]":
        from ..ops.repeat import Repeat
        
        return stage(Repeat[self.ST, self.SRK, RankGen().get_rank(int(self.SRK.__name__[1:])+1)](self, other), ctx=SrcCtx.new(2))
    
    # def flatmap(self, func: Callable[[A], Stream[B, b]]) -> "Stream[ST,SRK]":
    #     from ..ops.flatmap import Flatmap
        
    #     return stage(Flatmap[self.SRK, b, self.ST, B](self, func), ctx=SrcCtx.new(2))
    
    def partition(self, N: int, other: "RStream[ST,ST2,SRK,SRK2]") -> "Stream[ST,SRK-SRK2+1]":
        from ..ops.partition import Partition
        
        d = RankGen().get_rank(int(self.SRK.__name__[1:]) - int(self.SRK2.__name__[1:]) + 1)
        return stage(Partition[self.ST, self.ST2, self.SRK, self.SRK2, d](self, N, other), ctx=SrcCtx.new(2))


class HStream[ST,ST2,SRK](Ref[List[Union[Val[ST], Stop]], "HStream[ST,ST2,SRK]"]):
    @override
    def fresh(self) -> "Stream[ST,ST2,SRK]":
        return HStream[self.ST, self.ST2, self.SRK]()
    
    def map(self, func: Callable[[ST], ST2]) -> "HStream[ST2,SRK]":
        from ..ops.map import Map
        
        return stage(Map[self.ST, self.ST2, self.SRK](self, func), ctx=SrcCtx.new(2))
    
    def zip(self, other: "HStream[ST,ST2,SRK]") -> "Stream[Tuple[ST, ST2], SRK]":
        from ..ops.zip import Zip
        
        return stage(Zip[self.ST, other.ST2, self.SRK](self, other), ctx=SrcCtx.new(2))


class Stream[ST,SRK](Ref[List[Union[Val[ST], Stop]], "Stream[ST,SRK]"]):    
    @override
    def fresh(self) -> "Stream[ST,SRK]":
        return Stream[self.ST, self.SRK]()
    
    def bufferize(self, a: int) -> "Stream[Buffer[ST, a], SRK+1]":
        from ..ops.bufferize import Bufferize
        
        return stage(Bufferize[self.ST, RankGen().get_rank(a), self.SRK, RankGen().get_rank(int(self.SRK.__name__[1:])-a)](self), ctx=SrcCtx.new(2))
    
    def promote(self, b: int) -> "Stream[ST,SRK+1]":
        from ..ops.promote import Promote
        
        return stage(Promote[self.ST, self.SRK, RankGen().get_rank(int(self.SRK.__name__[1:])+1)](self, b), ctx=SrcCtx.new(2))
    
    def reshape(self, L: Tuple[Index, ...], S: Tuple[int, ...]) -> "Stream[ST,SRK+L]":
        from ..ops.reshape import Reshape
        
        return stage(Reshape[self.ST, self.SRK, RankGen().get_rank(int(self.SRK.__name__[1:])+len(L))](self, L, S), ctx=SrcCtx.new(2))
    
    def flatten(self, L: Tuple[Index, ...]) -> "Stream[ST,SRK-L]":
        from ..ops.flatten import Flatten
        
        assert len(L) < int(self.SRK.__name__[1:]) - 1
        return stage(Flatten[self.SRK, self.ST, RankGen().get_rank(int(self.SRK.__name__[1:])-len(L))](self, L), ctx=SrcCtx.new(2))
    
    def enumerate(self, b: int) -> "Stream[Tuple[ST,Index],SRK]":
        from ..ops.enumerate import Enumerate
        
        return stage(Enumerate[self.ST, self.SRK](self, b), ctx=SrcCtx.new(2))
    
    # def window(self, N: Tuple[int], S: Tuple[int]) -> "Stream[ST,SRK]":
    #     from ..ops.window import Window
        
    #     return stage(Window[self.ST, self.SRK, N, S](self, N, S), ctx=SrcCtx.new(2))