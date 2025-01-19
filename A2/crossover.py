import random

def create_chill(child, items, counter_tasks_job):
    counters = [0 for _ in range(len(counter_tasks_job))]
    item_id = 0
    for (position, job_id) in enumerate(child):
        if job_id is not None:
            counters[job_id] += 1
    for (position, job_id) in enumerate(child):
        if job_id is None:
            filled = False
            while not filled:
                new_job_id = items[item_id]
                if counters[new_job_id] < counter_tasks_job[new_job_id]:
                    child[position] = new_job_id
                    counters[new_job_id] += 1
                    filled = True
                item_id += 1
    return child


def one_point_crossover(parent1, parent2, counter_tasks_job):
    size_child = len(parent1)
    crossover_point = random.randint(1, size_child - 1)

    child1 = [None for _ in range(size_child)]    
    child1[0:crossover_point] = parent1[0:crossover_point]
    items_child1 = parent2[crossover_point:]
    items_child1.extend(parent2[0:crossover_point])
    child1 = create_chill(child1, items_child1, counter_tasks_job)
    
    child2 = [None for _ in range(size_child)]
    child2[0:crossover_point] = parent2[0:crossover_point]
    items_child2 = parent1[crossover_point:]
    items_child2.extend(parent1[0:crossover_point])
    child2= create_chill(child2, items_child2, counter_tasks_job)
    
    return child1, child2


def two_point_crossover(parent1, parent2, counter_tasks_job):
    size_child = len(parent1)
    crossover_point1, crossover_point2 = sorted(random.sample(range(1, size_child-1), 2))

    child1 = [None for _ in range(size_child)]  
    child1[0:crossover_point1] = parent1[0:crossover_point1]
    child1[crossover_point2:] = parent1[crossover_point2:]
    items_child1 = parent2[crossover_point1:crossover_point2]
    items_child1.extend(parent2[0:crossover_point1])
    items_child1.extend(parent2[crossover_point2:])
    child1 = create_chill(child1, items_child1, counter_tasks_job)
    
    child2 = [None for _ in range(size_child)]
    child2[0:crossover_point1] = parent2[0:crossover_point1]
    child2[crossover_point2:] = parent2[crossover_point2:]
    items_child2 = parent1[crossover_point1:crossover_point2]
    items_child2.extend(parent1[0:crossover_point1])
    items_child2.extend(parent1[crossover_point2:])
    child2= create_chill(child2, items_child2, counter_tasks_job)
    
    return child1, child2
