# from pydantic.dataclasses import dataclass
# from pydantic import ConfigDict
# from typing import Union, Tuple, override, List, TypeVar, Any
# from argon.ref import Ref
# from rankgen import RankGen
# import numpy as np
# import numpy.typing as npt

# T = TypeVar("T", bound=np.generic, covariant=True)

# @dataclass(config=ConfigDict(arbitrary_types_allowed=True))
# class Ndarray[T]:
#     value: npt.NDArray[T]
    
# BT = TypeVar("BT")
# BRK = TypeVar("BRK")

# class Buffer[BT,BRK](Ref[Ndarray[BT], "Buffer[BT,BRK]"]):

#     @override
#     def fresh(self) -> "Buffer[BT,BRK]":
#         return Buffer[self.BT, self.BRK]()

#     @override
#     def const(self, c: Ndarray[BT]) -> "Buffer[BT,BRK]":
#         assert RankGen().get_rank(c.value.ndim) == self.BRK
#         return super().const(c)