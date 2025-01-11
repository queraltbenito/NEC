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
