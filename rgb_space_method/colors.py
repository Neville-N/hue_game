import numpy as np
import copy


class Colors:
    def __init__(self) -> None:
        self.grid = [
            [[1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.0, 0.0]],
            [[0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.5, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.5, 0.5], [0.0, 1.0, 0.0]],
        ]

        self.swap(0, 1, 2, 1)
        self.swap(1, 1, 1, 2)

    def neihgbours(self, row: int, column: int):
        neigbours = []
        if row > 0:
            neigbours.append(self.color(row - 1, column))
        if column > 0:
            neigbours.append(self.color(row, column - 1))
        if row < self.max_index:
            neigbours.append(self.color(row + 1, column))
        if column < self.max_index:
            neigbours.append(self.color(row, column + 1))

        return neigbours

    def color(self, row: int, column: int):
        return np.array(self.grid[row][column])

    def swap(self, row1: int, column1: int, row2: int, column2: int):
        self.grid[row1][column1], self.grid[row2][column2] = (
            self.grid[row2][column2],
            self.grid[row1][column1],
        )

    def solve(self):
        constant = 0

        while constant < 100:
            after_swap = copy.deepcopy(self)
            indices = self.rinc + self.rinc
            after_swap.swap(*indices)
            if self.total_line_length > after_swap.total_line_length:
                self.swap(*indices)
                constant = 100
                print(self.total_line_length)
            else:
                constant += 1

    @property
    def rinc(self):
        index = [0, 0]
        corners = [[0, 0], [0, 2], [2, 0], [2, 2]]

        while index in corners:
            row = np.random.randint(0, self.size)
            column = np.random.randint(0, self.size)
            index = [row, column]

        return index

    @property
    def max_index(self):
        return self.size - 1

    @property
    def size(self):
        return len(self.grid[0])

    @property
    def total_line_length(self):
        total_line_length = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                home = self.color(i, j)
                neighbours = self.neihgbours(i, j)
                for neighbour in neighbours:
                    total_line_length += np.linalg.norm(home - neighbour)
        return total_line_length
