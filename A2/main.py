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
    num_jobs, num_machines, tasks = read_data(path)
    
    print(num_jobs)
    print(num_machines)
    print(tasks)

    items = generate_chromosome_items(tasks)
    population_size = 1
    population = [generate_chromosome(items, num_jobs) for _ in range(population_size)]
    print(population)

    duration = compute_duration(population[0], num_jobs, num_machines, tasks)
    print(duration)
