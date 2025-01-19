import random 

def generate_chromosome_items(total_tasks):
    items = []
    counter_tasks_job = []
    for (job_id, job) in enumerate(total_tasks):
        counter_tasks_job.append(len(job))
        items.extend([job_id] * len(job))
    return counter_tasks_job, items


def generate_chromosome(items, num_jobs):
    # A chromosome is a random order of the list of items
    chromosome = items.copy()
    random.shuffle(chromosome)
    return chromosome
