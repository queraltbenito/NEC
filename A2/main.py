from chromosome_representation import generate_chromosome_items, generate_chromosome
from fitness import compute_duration

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
    path = "./dataset1.txt"
    j, m, tasks = read_data(path)
    
    print(j)
    print(m)
    print(tasks)

    items = generate_chromosome_items(tasks)
    population_size = 5
    population = [generate_chromosome(items) for _ in range(population_size)]
    print(population)

    duration = compute_duration(population[0], j, m, tasks)
    print(duration)
