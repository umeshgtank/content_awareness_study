from deap import creator, base, tools, algorithms
import random
import statistics
import numpy
import matplotlib.pyplot as plt
import pickle

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to integers sampled uniformly
#                      from the range [0,1] (i.e. 0 or 1 with equal
#                      probability)
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of 100 'attr_bool' elements ('genes')
toolbox.register("individual", tools.initRepeat, creator.Individual,
    toolbox.attr_bool, 800)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def sol_fitness(individual):
    indices = [i for i, x in enumerate(individual) if (x == 1)]
    ind_fitness_score = get_avg_score(indices)
    return ind_fitness_score,


def plot_evolution(logbook):
    gen = logbook.select("gen")
    #fit_mins = logbook.chapters["fitness"].select("min")
    fit_mins = logbook.chapters["fitness"].select("avg")
    #size_avgs = logbook.chapters["size"].select("avg")
    size_avgs = logbook.select("evals")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "g-", label="Average population evaluated")
    ax2.set_ylabel("Evaluated", color="g")
    for tl in ax2.get_yticklabels():
        tl.set_color("g")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()

def get_avg_score(indices):
    score_sum = 0
    score_count = 0
    for key in indices:
        score_sum = score_sum + float(prepx_score[str(key)][0])
        score_count = score_count+1
    return score_sum/score_count


prepx_score = {}

def get_best_individuals(best_ind, best_gen_file):
    index = 0
    with open(best_gen_file, "w") as o_file:
        for key in best_ind:
            if int(key) == 1:
                o_file.write(str(prepx_score[str(index)][0]) + "\t" + prepx_score[str(index)][1] + "\n")
            index = index+1


def load_prepx_score(score_file):
    index_counter = 0
    with open(score_file, 'r') as df:
        # next(data_file)  # Skipping the header
        for line in df:
            line_comp = line.split("\t")
            prepx_score[str(index_counter)] = (line_comp[0], line_comp[1])
            index_counter = index_counter + 1

#----------
# Operator registration
#----------
# register the goal / fitness function
toolbox.register("evaluate", sol_fitness)

# register the crossover operator
toolbox.register("crossover", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

if __name__ == '__main__':
    score_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_data_pred.txt"
    load_prepx_score(score_file)
    test_score = get_avg_score([1,2,3])

    # Logging data
    logbook = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

    mstats.register("avg", numpy.mean, axis=0)
    mstats.register("std", numpy.std, axis=0)
    mstats.register("min", numpy.min, axis=0)
    mstats.register("max", numpy.max, axis=0)

    random.seed(64)

    # create an initial population of 10000 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=10000)

    # record = ga_stats.compile(pop)
    # logbook.record(gen=0, evals=10000, **record)
    logbook.header = "gen", "avg", "spam"
    logbook.chapters["fitness"].header = "min", "avg", "max"
    logbook.chapters["size"].header = "min", "avg", "max"
    # CXPB  is the probability with which two individuals
    #       are crossed
    #
    # MUTPB is the probability for mutating an individual
    CXPB, MUTPB = 0.5, 0.2

    print("Start of evolution")

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % str(sum(fits) / len(pop)))
    print("  Population length %s" % str(len(pop)))

    # Variable keeping track of the number of generations
    g = 0

    # Begin the evolution
    while statistics.mean(fits) > 4 and g < 3: #< 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.crossover(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        record = mstats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        print(logbook.stream)

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

    print("-- End of (successful) evolution --")
    plot_evolution(logbook)
    result_file_name = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_best_gen_result.txt"
    result_file = open(result_file_name, 'wb')
    pickle.dump(logbook, result_file)
    print(logbook)
    result_file.close()
    result_chapter_file_name = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_best_gen_result_chapters.txt"
    chapter_file = open(result_chapter_file_name, 'wb')
    pickle.dump(logbook.chapters["fitness"], chapter_file)
    chapter_file.close()


    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    print("  Size of the best individual is %s" % str(len(best_ind)))
    best_gen_file = "/Users/lab/Umesh/Course/sem4/src/xformer/phase2/data/ga_best_gen.txt"
    get_best_individuals(best_ind, best_gen_file)