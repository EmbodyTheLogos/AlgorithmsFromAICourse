########################################################
#
# CMPSC 441: Homework 4
#
########################################################


student_name = 'Long Nguyen'
student_email = 'lhn5032@psu.edu'

########################################################
# Import
########################################################

from hw4_utils import *
import math
import random
import collections


# Add your imports here if used


################################################################
# 1. Genetic Algorithm
################################################################


def genetic_algorithm(problem, f_thres, ngen=1000):
    population = problem.init_population()
    best = problem.fittest(population, f_thres)
    if best is not None:
        return -1, best
    for i in range(ngen):
        population = problem.next_generation(population)
        best = problem.fittest(population, f_thres)
        if best is not None:
            return i, best
    best = problem.fittest(population)
    return ngen, best

    """
    Returns a tuple (i, sol) 
    where
      - i  : number of generations computed
      - sol: best chromosome found
    """



################################################################
# 2. NQueens Problem
################################################################


class NQueensProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        super().__init__(n, g_bases, g_len, m_prob)
        self.n = n
        self.g_bases = g_bases
        self.g_len = g_len
        self.m_prob = m_prob

    def randomChromosome(self):
        chromosome = [random.choice(self.g_bases) for k in range(self.g_len)]
        return tuple(chromosome)

    def init_population(self):
        listOfChromosomes = [self.randomChromosome() for i in range(self.n)]
        return listOfChromosomes

    def next_generation(self, population):
        nextGeneration = []
        for i in range(len(population)):
            index = random.randint(0, len(population) - 1)
            index1 = random.randint(0, len(population) - 1)
            if index1 == index:
                if index1 == len(population) - 1:
                    index1 -= 1
                else:
                    index1 += 1
            child = self.crossover(population[index], population[index1])
            child = self.mutate(child)
            nextGeneration.append(child)
        return nextGeneration

    def mutate(self, chrom):
        p = random.uniform(0,1)
        if p > self.m_prob:
            return chrom
        i = random.randint(0, self.g_len - 1)
        listChrom = list(chrom)
        newItem = random.choice(self.g_bases)
        listChrom[i] = newItem
        return tuple(listChrom)

    def crossover(self, chrom1, chrom2):
        i = random.randint(1, self.g_len - 2)
        listChrome1 = list(chrom1)
        listChrome2 = list(chrom2)
        return tuple(listChrome1[:i] + listChrome2[i:])

    def conflict(self, row1, col1, row2, col2):
        return row1 == row2 or col1 == col2 or abs(row1-row2) == abs(col1-col2)

    def fitness_fn(self, chrom):
        count = 0
        for i in range(len(chrom)-1):
            for k in range(i+1, len(chrom)):
                if not self.conflict(chrom[i], i, chrom[k], k):
                    count += 1
        return count

    def select(self, m, population):
        totalFitness = 0
        cummulativeProbablity = 0
        probablityDistribution = []
        cummulativeDistribution = []
        fitnesses = list(map(p.fitness_fn, population))
        for i in fitnesses:
            totalFitness += i
        for k in fitnesses:
            probablityDistribution.append(k/totalFitness)
        for l in probablityDistribution:
            cummulativeProbablity += l
            cummulativeDistribution.append(cummulativeProbablity)

        selectedChromosomes = []

        for i in range(m):
            prob = random.uniform(0,1)
            if prob <= cummulativeDistribution[0]:
                selectedChromosomes.append(population[0])
            for k in range(len(cummulativeDistribution) - 1):
                if cummulativeDistribution[k] < prob <= cummulativeDistribution[k + 1]:
                    selectedChromosomes.append(population[k+1])

        return selectedChromosomes

    def fittest(self, population, f_thres=None):
        fitnesses = list(map(p.fitness_fn, population))
        bestFitness = 0
        bestFitnessIndex = 0
        index = -1

        for i in fitnesses:
            index += 1
            if ( i > bestFitness):
                bestFitness = i
                bestFitnessIndex = index
        if f_thres is None or bestFitness >= f_thres:
            return population[bestFitnessIndex]
        else:
            return None





################################################################
# 3. Function Optimaization f(x,y) = x sin(4x) + 1.1 y sin(2y)
################################################################


class FunctionProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob):
        super().__init__(n, g_bases, g_len, m_prob)
        self.n = n
        self.g_bases = g_bases
        self.g_len = g_len
        self.m_prob = m_prob

    def randomChromosome(self):
        x = random.uniform(0, self.g_bases[0])
        y = random.uniform(0, self.g_bases[1])
        return x, y

    def init_population(self):
        return [self.randomChromosome() for i in range(self.n)]

    def next_generation(self, population):
        fitnesses = list(map(self.fitness_fn, population))
        chromosomeAndFitness = {fitnesses[i]: population[i] for i in range(len(population))}
        sortedChromosomes = dict(collections.OrderedDict(sorted(chromosomeAndFitness.items())))
        bestHalf = []
        bestHalfLength = int(len(population) / 2)

        for i in range(bestHalfLength):
            chromosome = sortedChromosomes.get(list(sortedChromosomes)[i])
            bestHalf.append(chromosome)
        otherHalfLength = int(len(population) - bestHalfLength)
        otherHalf = []
        for i in range(otherHalfLength):
            index1 = random.randint(0, bestHalfLength - 1)
            index2 = random.randint(0, bestHalfLength - 1)
            if index1 == index2:
                if index1 == len(bestHalf) - 1:
                    index1 -= 1
                else:
                    index1 += 1
            child = self.crossover(bestHalf[index1], bestHalf[index2])
            child = self.mutate(child)
            otherHalf.append(child)

        return bestHalf + otherHalf



    def mutate(self, chrom):
        listChromosome = list(chrom)
        p = random.uniform(0,1)
        if p > self.m_prob:
            return chrom
        i = random.randint(0, len(chrom) - 1)
        listChromosome[i] = random.uniform(0, self.g_bases[i])
        return tuple(listChromosome)


    def crossover(self, chrom1, chrom2):
        alpha = random.uniform(0,1)
        i = random.randint(0, 1)
        if i == 0:
            xNew = (1 - alpha) * chrom1[0] + alpha * chrom2[0]
            return xNew, chrom1[1]
        else:
            yNew = (1 - alpha) * chrom1[1] + alpha * chrom2[1]
            return chrom1[0], yNew

    def fitness_fn(self, chrom):
        x = chrom[0]
        y = chrom[1]
        fitness = x * math.sin(4 * x) + 1.1 * y * math.sin(2 * y)
        return fitness

    def select(self, m, population):
        summation = 0
        probabilityDistrinution = []
        cummulativeDistribution = []
        cummulativeProbability = 0

        for i in range(1, len(population)+ 1):
            summation += i
        rankFitness = list(map(self.fitness_fn, population))
        chromosomeAndFitness = {rankFitness[i]:population[i] for i in range(len(population))}
        sortedChromosomes = dict(collections.OrderedDict(sorted(chromosomeAndFitness.items())))

        for k in range(len(population)):
            probabilityDistrinution.append((len(population)-k)/summation)

        for l in probabilityDistrinution:
            cummulativeProbability += l
            cummulativeDistribution.append(cummulativeProbability)

        selectedChromosomes = []
        for i in range(m):
            prob = random.uniform(0,1)
            if prob <= cummulativeDistribution[0]:
                chromosome = sortedChromosomes.get(list(sortedChromosomes)[0])
                selectedChromosomes.append(chromosome)
            for k in range(len(cummulativeDistribution) - 1):
                if cummulativeDistribution[k] < prob <= cummulativeDistribution[k + 1]:
                    chromosome = sortedChromosomes.get(list(sortedChromosomes)[k+1])
                    selectedChromosomes.append(chromosome)

        return selectedChromosomes



    def fittest(self, population, f_thres=None):
        fitnesses = list(map(self.fitness_fn, population))
        bestFitness = fitnesses[0]
        bestFitnessIndex = 0
        index = -1

        for i in fitnesses:
            index += 1
            if (i < bestFitness):
                bestFitness = i
                bestFitnessIndex = index
        if f_thres is None or bestFitness <= f_thres:
            return population[bestFitnessIndex]
        else:
            return None



################################################################
# 4. Traveling Salesman Problem
################################################################


class HamiltonProblem(GeneticProblem):
    def __init__(self, n, g_bases, g_len, m_prob, graph=None):
        super().__init__(n, g_bases, g_len, m_prob)
        self.n = n
        self.g_bases = g_bases
        self.g_len = g_len
        self.m_prob = m_prob
        self.graph = graph

    def init_population(self):
        return [random.sample(self.g_bases, len(self.g_bases)) for i in range(self.n)]

    def new_generation(self, population):
        newGeneration = []
        for i in range(len(population)):
            index1 = random.randint(0, len(population) - 1)
            index2 = random.randint(0, len(population) - 1)
            if index1 == index2:
                if index1 == len(population) - 1:
                    index1 -= 1
                else:
                    index1 += 1
            newGeneration.append(self.crossover(population[index1], population[index2]))
        return newGeneration

    def next_generation(self, population):
        totalChromosomes = population + self.new_generation(population)

        fitnesses = list(map(self.fitness_fn, totalChromosomes))

        chromosomeAndFitness = {fitnesses[i]:totalChromosomes[i] for i in range(len(totalChromosomes))}

        #sort chromosomes according to fitness
        sortedChromosomes = dict(collections.OrderedDict(sorted(chromosomeAndFitness.items())))
        # dictionary removes duplicates keys

        newGeneration = []
        for i in range(len(population)):
            chromosome = sortedChromosomes.get(list(sortedChromosomes)[i])
            newGeneration.append(chromosome)
        return newGeneration


    def mutate(self, chrom):
        p = random.uniform(0,1)
        listChromosome = list(chrom)
        if p > self.m_prob:
            return chrom
        else:
            i = random.randint(0, len(chrom) - 1)
            k = random.randint(0, len(chrom) - 1)
            temp = listChromosome[i]
            listChromosome[i] = listChromosome[k]
            listChromosome[k] = temp
            return tuple(listChromosome)



    def crossover(self, chrom1, chrom2):
        offSpring1 = list(chrom1)
        offSpring2 = list(chrom2)
        i = random.randint(0, len(offSpring1) - 1)

        temp = offSpring1[i]
        offSpring1[i] = offSpring2[i]
        offSpring2[i] = temp

        while set(offSpring1) != set(chrom1):
            for k in range(len(offSpring1)):
                if k != i and offSpring1[k] == offSpring1[i]:
                    i = k
                    temp = offSpring1[k]
                    offSpring1[k] = offSpring2[k]
                    offSpring2[k] = temp

        return tuple(offSpring1)

    def fitness_fn(self, chrom):
        distance = 0
        for i in range(len(chrom)-1):
            distance += self.graph.get(chrom[i], chrom[i+1])
        return distance + self.graph.get(chrom[len(chrom) - 1], chrom[0])

    def select(self, m, population):
        T = 0
        fitnesses = list(map(p.fitness_fn, population))
        chromosomeProbability = 0
        cummulativeDistribution = []
        denominatorSumation = 0

        for i in fitnesses:
            T += i

        for j in range(len(population)):
            denominatorSumation += (T - fitnesses[j])

        for k in range(len(population)):
            chromosomeProbability += (T - fitnesses[k]) / denominatorSumation
            cummulativeDistribution.append(chromosomeProbability)

        selectedChromosomes = []

        for i in range(m):
            prob = random.uniform(0,1)
            if prob <= cummulativeDistribution[0]:
                selectedChromosomes.append(population[0])
            for k in range(len(cummulativeDistribution) - 1):
                if cummulativeDistribution[k] < prob <= cummulativeDistribution[k + 1]:
                    selectedChromosomes.append(population[k + 1])

        return selectedChromosomes

    def fittest(self, population, f_thres=None):
        fitnesses = list(map(self.fitness_fn, population))
        bestFitness = fitnesses[0]
        bestFitnessIndex = 0
        index = -1

        for i in fitnesses:
            index += 1
            if (i < bestFitness):
                bestFitness = i
                bestFitnessIndex = index
        if f_thres is None or bestFitness <= f_thres:
            return population[bestFitnessIndex]
        else:
            return None
