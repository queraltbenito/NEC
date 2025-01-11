def elitism(population, N):
    population_sorted = sorted(population, key=lambda x: x[1], reverse=True)
    return population_sorted[:N]
