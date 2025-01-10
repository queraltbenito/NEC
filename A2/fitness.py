def machine_item(job_id, task_id, tasks):
    return int(tasks[job_id][task_id][0])

def processing_time_item(job_id, task_id, tasks):
    return int(tasks[job_id][task_id][1])

def compute_duration(chromosome, num_jobs, num_machines, tasks):
    f = 0
    completition_jobs = [0 for _ in range(num_jobs)]
    availability_machines = [0 for _ in range(num_machines)]
    
    for item in chromosome:
        job_id, task_id = item.split('-')
        job_id = int(job_id)
        task_id = int(task_id)

        m = machine_item(job_id, task_id, tasks)
        p = processing_time_item(job_id, task_id, tasks)
        print(completition_jobs, availability_machines, item, m, p, f)

        t = max(completition_jobs[job_id], availability_machines[m]) + p
        completition_jobs[job_id] = t
        availability_machines[m] = t
        f = max(f, t)

    return f
