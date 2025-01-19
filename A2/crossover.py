import random
from collections import Counter

def one_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:crossover_point]
    child2 = parent2[:crossover_point]
    
    #### FALTA
    return child1, child2

def two_point_crossover(parent1, parent2):
    child1 = parent1
    child2 = parent2
    return child1, child2
