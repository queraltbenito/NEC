import random 

def generate_chromosome_items(total_tasks):
    items = []
    for (job_id, job) in enumerate(total_tasks):
        items.extend([job_id] * len(job))
    return items


def generate_chromosome(items, num_jobs):
    # A chromosome is a random order of the list of items
    chromosome = items.copy()
    random.shuffle(chromosome)
    return chromosome
