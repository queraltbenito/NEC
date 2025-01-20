import random

def roulette_wheel_selection(population):
    # Calculate the total fitness
    total_fitness = sum(fitness for _, fitness in population)

    # Generate a random number
    pick = random.uniform(0, total_fitness)
    
    # Traverse through the population and select the chromosome
    current_sum = 0
    for chromosome, fitness in population:
        current_sum += fitness
        if current_sum > pick:
            return chromosome


def rank_selection(population):
    # Sort the population by fitness in ascending order (lower rank -> higher fitness)
    ranked_population = sorted(population, key=lambda x: x[1])

    # Assign ranks (1 for the least fit, len(population) for the fittest)
    ranks = [i + 1 for i in range(len(ranked_population))]

    # Calculate the total rank sum
    total_rank = sum(ranks)

    # Calculate selection probabilities based on ranks
    probabilities = [rank / total_rank for rank in ranks]

    # Perform selection using cumulative probabilities
    cumulative_probabilities = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    pick = random.uniform(0, 1)

    for i, cp in enumerate(cumulative_probabilities):
        if pick <= cp:
            return ranked_population[i][0]
