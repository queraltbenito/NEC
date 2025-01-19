import random 

def one_mutation(chromosome):
    if len(chromosome) < 2: return chromosome
    
    id1, id2 = random.sample(range(len(chromosome)), 2)
    chromosome[id1], chromosome[id2] = chromosome[id2], chromosome[id1]
    return chromosome

def prob_mutation():
    return
