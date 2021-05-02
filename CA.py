import numpy as np
import math
import random

class CA:
    def __init__(self, w, h):
        self.cells = np.zeros((w, h))
        self.width = w
        self.height = h
        self.generations = []

        for i in range(h): 
            self.cells[i, w - 1] = i * (h - i)
        
        self.generations.append(self.cells)

    def set_cell(self, x, y, val):
        self.cells[x, y] = val
  
    def get_cells(self):
        return self.cells

    def generate(self):
        h = self.height
        w = self.width
        nextgen = np.array(self.cells)

        for row in range(1, h - 1):
            for col in range(1, w - 1):
                left = self.cells[row, col - 1]
                right = self.cells[row, col + 1]
                up = self.cells[row - 1, col]
                down = self.cells[row + 1, col]

                nextgen[row, col] = self.update([up, down, left, right])

        self.cells = nextgen[::]
        self.generations.append(self.cells)

    def update(self, neighbors):
        return sum(neighbors) / 4

    def Sum(self, cells):
        return sum([sum(row) for row in cells])
    
    def check_for_stability(self):
        if abs(self.Sum(self.generations[-1]) - self.Sum(self.generations[-2])) > 0:
            return False
        return True