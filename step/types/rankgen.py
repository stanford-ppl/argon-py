from typing import Any


class RankGen:
    generated_ranks = {}

    def get_rank(self, c: int) -> Any:
        if str(c) not in self.generated_ranks:
            self.generated_ranks[str(c)] = type("R" + str(c), (), {})
        return self.generated_ranks[str(c)]
