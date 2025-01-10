import random 

def generate_chromosome_items(total_tasks):
    items = []
    for (job_id, job) in enumerate(total_tasks):
        for (task_id, task) in enumerate(job):
            items.append(f"{job_id}-{task_id}")
    return items

def valid_chromosome(chromosome):
    return True

def generate_chromosome(items):
    # A chromosome is a random order of the list of items
    chromosome = items.copy()
    random.shuffle(chromosome)
    while not valid_chromosome(chromosome):
        random.shuffle(chromosome)
    return chromosome
