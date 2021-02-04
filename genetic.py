import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

mute_rate = 0.02
cross_rate = 0.9
n_cities = 29
n_population = 200
epoch = 100
mean, best, worst = [], [], []


class Pop:
    def __init__(self, pop, distance):
        self.pop = pop
        self.distance = distance


def fitness(x):
    fit = np.zeros(len(x.pop))
    for i in range(len(x.pop)):
        for j in range(len(x.pop[i]) - 1):
            fit[i] += x.distance[x.pop[i][j], x.pop[i][j + 1]]

    return fit


def select(x):
    t = 3
    fit = fitness(x)
    fit = np.max(fit) - fit
    new_pop = []
    for i in range(len(x.pop) - 1):
        rand = np.random.choice(len(x.pop), size=t, replace=False)
        best, index = 0, 0
        # Find the best one in t picked chromosomes
        for j in rand:
            if fit[j] > best:
                best = fit[j]
                index = j

        new_pop.append(x.pop[index])
    new_pop = np.asarray(new_pop)
    return new_pop


def crossover(x):
    global cross_rate
    new_pop = []
    rand = np.random.choice(len(x.pop), size=len(x.pop), replace=False)
    for k in range(0, len(rand), 2):
        # Two parent from a random sequence and create two child with probability of cross_rate
        # If there is only one parent left, move it to the next generation
        a = x.pop[rand[k]].copy()
        if k + 1 < len(rand):
            # more than one parents left
            b = x.pop[rand[k + 1]].copy()
            new_a = a.copy()
            new_b = b.copy()
            prob = random.uniform(0, 1)
            if not np.array_equal(a, b) and prob <= cross_rate:
                # print('parent', a, b)
                points = len(a)
                turn = 0
                while points > 0:
                    turn += 1
                    aa = [a[0]]
                    bb = [b[0]]
                    # x1 pointer in first parent, x2 in second parent
                    x1, x2 = a[0], b[0]
                    # Check if the turn is even for starting points so we need to crossover them
                    # print(turn, x1, x2)
                    if turn % 2 == 0:
                        for j in range(len(x.pop[rand[k]])):
                            if x1 == x.pop[rand[k]][j]:
                                new_a[j] = x2
                                new_b[j] = x1
                                break
                    while x2 not in aa:
                        for i in range(len(a)):
                            # find new x1-->x2
                            if a[i] == x2:
                                x1 = a[i]
                                x2 = b[i]
                                aa.append(x1)
                                bb.append(x2)
                                points = points - 1
                                break
                        # Check if the turn is even for two points so we need to crossover them
                        # print(turn, x1, x2)
                        if turn % 2 == 0:
                            for j in range(len(x.pop[rand[k]])):
                                if x1 == x.pop[rand[k]][j]:
                                    new_a[j] = x2
                                    new_b[j] = x1
                                    break
                    a = np.array([el for el in a if el not in aa])
                    b = np.array([el for el in b if el not in bb])
                    points = len(a)
                # print('child', new_a, new_b)
            new_pop.append(new_a)
            new_pop.append(new_b)
        else:
            # one parent left
            new_pop.append(a)

    new_pop = np.asarray(new_pop)
    return new_pop


def mutation(x):
    global mute_rate
    for k in range(len(x.pop)):
        prob = random.uniform(0, 1)
        if prob >= (1 - mute_rate):
            a = x.pop[k].copy()
            # print('origin', a)
            new_a = a.copy()
            # Two random gene
            rand = np.random.choice(len(x.pop[0]), size=2, replace=False)
            rand = np.sort(rand)
            i = rand[0]
            new_a[i + 1] = a[rand[1]]
            shift = []
            for j in range(i + 1, len(a)):
                if j != rand[1]:
                    shift.append(a[j])
            # Shift
            for j in range(len(shift)):
                new_a[j + i + 2] = shift[j]
            # print('mutant', new_a)
            x.pop[k] = new_a.copy()
    return x


# Reading data and defining cities
cities = np.arange(0, n_cities-1, 1)
distance = np.zeros((n_cities, n_cities))
data = pd.read_csv('bayg29.csv', header=None)
for i in range(n_cities):
    for j in range(i * (n_cities - 1), (i + 1) * (n_cities - 1)):
        distance[i, data.iloc[j, 0]] = data.iloc[j, 1]
distance = np.asarray(distance)

# First random population
first = []
for i in range(n_population):
    first.append(np.random.permutation(cities))
print('first population')
first = np.asarray(first)
population = Pop(first, distance)
print(population.pop)
mean.append(np.sum(fitness(population)/ len(population.pop)))
best.append(np.min(fitness(population)))
worst.append(np.max(fitness(population)))
for i in range(epoch):
    print('Epoch %s' % i)
    # print('select')
    population.pop = select(population)
    # print(population.pop)
    # print('crossover')
    population.pop = crossover(population)
    # print(population.pop)
    # print('mutation')
    population = mutation(population)
    # print(population.pop)
    mean.append(np.sum(fitness(population) / len(population.pop)))
    best.append(np.min(fitness(population)))
    worst.append(np.max(fitness(population)))

print(best[-1])
e = np.arange(0, epoch+1, 1)
plt.plot(e, mean, 'b', e, worst, 'r', e, best, 'g')
plt.ylabel('Distance')
plt.xlabel('Epoch')
plt.legend(['Mean', 'Worst', 'Best'])
plt.show()