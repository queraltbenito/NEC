from chromosome_representation import generate_chromosome_items, generate_chromosome
from fitness import compute_duration
from selection import roulette_wheel_selection, stochastic_universal_sampling
from crossover import one_point_crossover, two_point_crossover
from mutation import one_mutation, prob_mutation
from elitism import elitism

def read_data(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    j, m = map(int, lines[0].strip().split())
    
    tasks = []
    for line in lines[1:]: 
        tasks_job = list(map(int, line.strip().split()))
        pairs = [(tasks_job[i], tasks_job[i+1]) for i in range(0, len(tasks_job), 2)]
        tasks.append(pairs)

    return j, m, tasks

if __name__ == "__main__":
    # Input data
    path = "./dataset1.txt"
    num_jobs, num_machines, tasks = read_data(path)
    
    print(num_jobs)
    print(num_machines)
    print(tasks)

    # Initialize population P
    # Evaluate fitness of all individuals in P
    items = generate_chromosome_items(tasks)
    population_size = 10
    population = []
    for _ in range(population_size):
        chromosome = generate_chromosome(items, num_jobs)
        duration = compute_duration(chromosome, num_jobs, num_machines, tasks)
        population.append((chromosome, 1/duration))

    # Genetic Algorithm
    num_generation = 2
    for generation in range(num_generation):
        population_size = len(population)
        new_population = []
        for pair in range(int(population_size/2)):
            # 1) Define new chromosomes
            # Selection process
            parent1 = roulette_wheel_selection(population)
            parent2 = roulette_wheel_selection(population)
            # Crossover process
            child1, child2 = one_point_crossover(parent1, parent2) 
            # Mutation process
            mutated_child1 = one_mutation(child1)
            mutated_child2 = one_mutation(child2)

            # 2) Compute new fitness
            duration1 = compute_duration(mutated_child1, num_jobs, num_machines, tasks)
            duration2 = compute_duration(mutated_child2, num_jobs, num_machines, tasks)

            # 3) Add new element to the population
            new_population.append((mutated_child1, 1/duration1))
            new_population.append((mutated_child2, 1/duration2))

        
        # Elitism. Add best fitted individuals of P to P'
        new_population.extend(elitism(population, 5))

        population = new_population
    
    best_item = elitism(population, 1)[0]
    best_chromosome = best_item[0]
    best_duration = 1/best_item[1]
    print(best_chromosome, best_duration)
