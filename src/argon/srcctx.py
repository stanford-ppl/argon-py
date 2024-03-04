import inspect
import typing
import dis
from pydantic.dataclasses import dataclass

@dataclass(unsafe_hash=True, frozen=True, slots=True)
class SrcCtx:
    file: str
    positions: dis.Positions | None

    @staticmethod
    def new(depth: int = 1) -> 'SrcCtx':
        """Captures the source context, which allows later reference for debugging.

        Args:
            depth : int
                The capture depth. Depth=1 captures the caller of new(), while greater depths enable
                capturing further outside.
        """
        frame = inspect.stack()[depth]
        return SrcCtx(frame.filename, frame.positions)

