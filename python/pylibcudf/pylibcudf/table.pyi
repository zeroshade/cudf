# Copyright (c) 2024, NVIDIA CORPORATION.

from pylibcudf.column import Column

class Table:
    def __init__(self, column: list[Column]): ...
    def num_columns(self) -> int: ...
    def num_rows(self) -> int: ...
    def columns(self) -> list[Column]: ...