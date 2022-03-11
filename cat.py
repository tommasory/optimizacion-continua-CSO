import numpy as np
from solution import solution
from enum import Enum

class Behavior(Enum):
    SEEKING = 1
    TRACING = 2

class cat(solution):

    def __init__(self, d: int, f, behavior=None):
        super().__init__(d,f)
        self.behavior = behavior
        self.velocity = np.zeros(self.size, float)
        self.bestcells = np.zeros(self.size, float)
        self.bestfitness = 0.0

    def from_cat(self, origin):
        super().from_solution(origin)

        self.velocity = np.copy(origin.velocity)
        self.bestcells = np.copy(origin.bestcells)
        self.bestfitness = origin.bestfitness
        
    def Initialization(self):
        self.cells = np.random.uniform(low=self.function.lower_bound, high=self.function.upper_bound,size=(self.size,))
        
        self.fitness = self.function.evaluate(self.cells)

        self.velocity = np.random.uniform(low=-4, high=4,size=(self.size,))
        self.bestcells = np.copy(self.cells)
        self.bestfitness = self.fitness

    def move(self, beta, delta, alfa, epsilon, best_cells):

        for dim in range(self.size):
            my_beta = np.random.uniform(low=0, high=beta,size=1)[0]
            my_delta = np.random.uniform(low=0, high=delta, size=1)[0]

            self.velocity[dim] = alfa * self.velocity[dim] + my_beta * (self.bestcells[dim] - self.cells[dim]) + my_delta * (best_cells[dim] - self.cells[dim])

        self.cells = self.cells + epsilon * self.velocity

        self.cells[self.cells < self.function.lower_bound] = self.function.lower_bound
        self.cells[self.cells > self.function.upper_bound] = self.function.upper_bound

        self.fitness = self.function.evaluate(self.cells)

    def __str__(self):
        return "cells:" + str(self.cells) + \
               "-fit:" + str(self.fitness)
