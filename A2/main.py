import matplotlib.pyplot as plt
from chromosome_representation import generate_chromosome_items, generate_chromosome
from fitness import compute_duration
from selection import roulette_wheel_selection, rank_selection
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
    path = "./dataset3.txt"
    num_jobs, num_machines, tasks = read_data(path)
    
    print("\nNumber of Jobs:", num_jobs)
    print("Number of Machines:", num_machines)
    print("Tasks:", tasks)

    # User input for choices
    print("Select Selection Method:")
    print("1: Roulette Wheel Selection")
    print("2: Rank Selection")
    selection_choice = input("Enter 1 or 2: ")
    selection_function = roulette_wheel_selection if selection_choice == "1" else rank_selection

    print("\nSelect Crossover Method:")
    print("1: One-Point Crossover")
    print("2: Two-Point Crossover")
    crossover_choice = input("Enter 1 or 2: ")
    crossover_function = one_point_crossover if crossover_choice == "1" else two_point_crossover

    print("\nSelect Mutation Method:")
    print("1: One Mutation")
    print("2: Probability Mutation")
    mutation_choice = input("Enter 1 or 2: ")
    mutation_function = one_mutation if mutation_choice == "1" else prob_mutation

    # Initialize population
    counter_tasks_job, items = generate_chromosome_items(tasks)
    population_size = 15
    population = []
    for _ in range(population_size):
        chromosome = generate_chromosome(items, num_jobs)
        duration = compute_duration(chromosome, num_jobs, num_machines, tasks)
        population.append((chromosome, 1/duration))

    # Genetic Algorithm
    num_generation = 100  # Set number of generations
    min_fitness_history = []  # To track minimum fitness in each generation

    for generation in range(num_generation):
        population_size = len(population)
        new_population = []
        for pair in range(int(population_size / 2)):
            # Selection process
            parent1 = selection_function(population)
            parent2 = selection_function(population)
            # Crossover process
            child1, child2 = crossover_function(parent1, parent2, counter_tasks_job)
            # Mutation process
            mutated_child1 = mutation_function(child1)
            mutated_child2 = mutation_function(child2)

            # Compute fitness
            duration1 = compute_duration(mutated_child1, num_jobs, num_machines, tasks)
            duration2 = compute_duration(mutated_child2, num_jobs, num_machines, tasks)

            # Add new elements to the population
            new_population.append((mutated_child1, 1/duration1))
            new_population.append((mutated_child2, 1/duration2))

        # Elitism. Add best-fitted individuals of P to P'
        new_population.extend(elitism(population, 5))

        population = new_population

        # Record the minimum fitness in this generation
        min_fitness = min(1 / individual[1] for individual in population)
        min_fitness_history.append(min_fitness)

    # Get the best solution
    best_item = elitism(population, 1)[0]
    best_chromosome = best_item[0]
    best_duration = 1 / best_item[1]
    print("\nBest Chromosome:", best_chromosome)
    print("Best Duration:", best_duration)

    # Plot the evolution of the minimum fitness
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_generation), min_fitness_history, label="Minimum Traveling Distance")
    plt.xlabel("Generation")
    plt.ylabel("Minimum Traveling Distance")
    plt.title("Evolution of Minimum Traveling Distance Across Generations")
    plt.legend()
    plt.grid()
    plt.show()
