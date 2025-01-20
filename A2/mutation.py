import random 

def one_mutation(chromosome):
    if len(chromosome) < 2: return chromosome
    
    id1, id2 = random.sample(range(len(chromosome)), 2)
    chromosome[id1], chromosome[id2] = chromosome[id2], chromosome[id1]
    return chromosome

def prob_mutation(chromosome, mutation_prob=0.1):
    for i in range(len(chromosome)):
        if random.random() < mutation_prob:
            # Select a random index to swap with
            swap_idx = random.randint(0, len(chromosome) - 1)
            # Swap the two genes
            chromosome[i], chromosome[swap_idx] = chromosome[swap_idx], chromosome[i]
    return chromosome
