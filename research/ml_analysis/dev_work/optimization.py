"""
Module to define series of optimization algorithms
"""
# import pdb
import time
import random
import math

PEOPLE = [('Seymour', 'BOS'),
          ('Franny', 'DAL'),
          ('Zooey', 'CAK'),
          ('Walt', 'MIA'),
          ('Buddy', 'ORD'),
          ('Les', 'OMA')]

# Need to get everyone from their origin to the below destination
# and then will return to their origin on the same day
DESTINATION = 'LGA'
FLIGHTS = {}

with open('schedule.txt') as file:
    LINES = file.readlines()
    for line in LINES:
        origin, dest, depart, arrive, price = line.strip().split(',')
        FLIGHTS.setdefault((origin, dest), [])
        # Add details to the list of possible flights
        FLIGHTS[(origin, dest)].append((depart, arrive, int(price)))


def get_mins(t_time):
    """
    calculates how many minutes into the day a given time is
    """
    mins = time.strptime(t_time, '%H:%M')
    return mins[3] * 60 + mins[4]


def print_sched(sol):
    """
    prints the output of the flights ppl decide to take
    """
    for ind in range(int(len(sol) / 2)):
        name = PEOPLE[ind][0]
        orig = PEOPLE[ind][1]
        out = FLIGHTS[(orig, DESTINATION)][int(sol[ind])]
        ret = FLIGHTS[(DESTINATION, orig)][int(sol[ind+1])]
        print('%10s%10s %5s-%5s $%3s %5s-%5s $%3s' % (name, orig,
                                                      out[0], out[1], out[2],
                                                      ret[0], ret[1], ret[2]))


def sched_cost(sol):
    """
    Cost function for the flight schedule
    """
    totalprice = 0
    latest_arrival = 0
    earliest_dep = 24 * 60

    for ind in range(int(len(sol) / 2)):
        # Get the inbound and outbound flights
        orig = PEOPLE[ind][1]
        # what outbound flight this person is taking
        outbound = FLIGHTS[(orig, DESTINATION)][int(sol[ind])]
        # what return flight this person is taking
        returnf = FLIGHTS[(DESTINATION, orig)][int(sol[ind+1])]

        # Total price is the price of all outbound and return flights
        totalprice += outbound[2]
        totalprice += returnf[2]

        # Track the latest arrival and earliest departure
        if latest_arrival < get_mins(outbound[1]):
            latest_arrival = get_mins(outbound[1])
        if earliest_dep > get_mins(returnf[0]):
            earliest_dep = get_mins(returnf[0])

    # Every person must wait at the airport until the latest person arrives.
    # They also must arrive at the same time and wait for their flights.
    totalwait = 0
    for ind in range(int(len(sol)/2)):
        orig = PEOPLE[ind][1]
        outbound = FLIGHTS[(orig, DESTINATION)][int(sol[ind])]
        returnf = FLIGHTS[(DESTINATION, orig)][int(sol[ind+1])]
        # Time waiting for last person to land
        totalwait += latest_arrival - get_mins(outbound[1])
        # Time waiting for return flight to take off
        totalwait += get_mins(returnf[0]) - earliest_dep

    # Does this solution require an extra day of car rental? That'll be $50!
    if latest_arrival > earliest_dep:
        totalprice += 50
    return totalprice + totalwait


def random_opt(domain, costf):
    """
    randomly try solutions and pick the best one
    """
    best = 999999999
    bestr = None
    for _ in range(0, 1000):
        # Create a random solution
        ret = [float(random.randint(domain[ind][0], domain[ind][1]))
               for ind in range(len(domain))]

        # Get the cost
        cost = costf(ret)

        # Compare it to the best one so far
        if cost < best:
            best = cost
            bestr = ret
    return bestr


def hill_climb_opt(domain, costf):
    """
    Hill Climb optimization
    Will look to left and right and continue down the slope to better solutions
    vulnerable to local minima
    """
    # Create a random solution
    sol = [random.randint(domain[i][0], domain[i][1])
           for i in range(len(domain))]

    # Main loop
    while 1:
        # Create list of neighboring solutions
        neighbors = []

        for ind, dom in enumerate(domain):
            # Loop through each input, add 1 and subtract 1 from each
            # add each permutation to neighbors
            if sol[ind] > dom[0] and sol[ind] != 9:
                neighbors.append(sol[0:ind] + [sol[ind]+1] + sol[ind+1:])
            if sol[ind] < dom[1] and sol[ind] != 0:
                neighbors.append(sol[0:ind] + [sol[ind]-1] + sol[ind+1:])

        # See what the best solution amongst the neighbors is
        current = costf(sol)
        best = current
        for nbr in neighbors:
            cost = costf(nbr)
            if cost < best:
                best = cost
                sol = nbr

        # If there's no improvement, then we've reached the top
        if best == current:
            break
    return sol


def anneal_opt(domain, costf, temp=10000.0, cool=0.95, step=2):
    """
    Annealing solution that will accept all better solutions but also randomly
    accept worse solutions
    Probability of accepting worse solutions decreases with time
    """
    # Initialize the values randomly
    vec = [float(random.randint(domain[i][0], domain[i][1]))
           for i in range(len(domain))]

    while temp > 0.1:
        # Choose one of the indices
        ind = random.randint(0, len(domain)-1)

        # Choose a direction to change it
        direction = random.randint(-step, step)
        # Create a new list with one of the values changed
        vecb = vec
        vecb[ind] += direction
        # Need this incase the new value is outside the domain
        if vecb[ind] < domain[ind][0]:
            vecb[ind] = domain[ind][0]
        elif vecb[ind] > domain[ind][1]:
            vecb[ind] = domain[ind][1]

        # Calculate the current cost and the new cost
        old_cost = costf(vec)
        new_cost = costf(vecb)
        # calculate the hurdle for randomly accepting a poorer solution
        # hurdle will get harder and harder to pass as temperature decreases
        hurdle = pow(math.e, (-new_cost - old_cost) / temp)

        # new solution is better
        # or hurdle for accepting worse solution is reached
        if (new_cost < old_cost or random.random() < hurdle):
            vec = vecb

        # Decrease the temperature
        temp = temp * cool
        # print(temp)
    return vec


def genetic_opt(domain, costf, popsize=50, step=1, mutprob=0.2, elite=0.2, maxiter=100):
    """
    Takes a random population and successively keeps the best members
    and creates new members from the best that have survived
    """
    # Build the initial population of solution
    pop = []
    for _ in range(popsize):
        vec = [random.randint(domain[ind][0], domain[ind][1])
               for ind in range(len(domain))]
        pop.append(vec)

    # only select elite winners from each generation
    topelite = int(elite * popsize)

    # Main loop
    for _ in range(maxiter):
        # rank the costs of the population
        scores = [(costf(v), v) for v in pop]
        scores.sort()
        ranked = [v for (s, v) in scores]

        # Keep the elite winners
        pop = ranked[0:topelite]

        while len(pop) < popsize:
            # Add mutated and bred forms of the winners based on mutation prob
            # Add mutated forms of the winners based on mutation prob
            if random.random() < mutprob:
                # Mutation
                ind = random.randint(0, topelite)
                pop.append(mutate(ranked[ind], domain, step))
            else:
                # Otherwise add Crossovers by combining two of the elite members
                vec1 = random.randint(0, topelite)
                vec2 = random.randint(0, topelite)
                pop.append(crossover(ranked[vec1], ranked[vec2], domain))
        # Print current best score
        # print(scores[0][0])
    return scores[0][1]


# Mutation Operation
def mutate(vec, domain, step):
    """
    Take a solution and mutate it
    """
    # randomly select and index
    index = random.randint(0, len(domain)-1)
    # randomly choose to move it up or down by a step
    # Check to make sure potential new values are in the domain
    if random.random() < 0.5 and vec[index] - step >= domain[index][0]:
        return vec[0:index] + [vec[index]-step] + vec[index+1:]
    elif vec[index] + step <= domain[index][1]:
        return vec[0:index] + [vec[index]+step] + vec[index+1:]
    # return vec if mutation puts an input outside domain range
    return vec


# Crossover Operation
def crossover(vec1, vec2, domain):
    """
    combine two vectors at a random point to "cross-breed"
    """
    ind = random.randint(1, len(domain)-2)
    return vec1[0:ind] + vec2[ind:]


if __name__ == '__main__':
    # test printschedule functionality
    # TEMP_SOL = [1,4,3,2,7,3,6,3,2,4,5,3]
    # print_sched(TEMP_SOL)
    # print(sched_cost(TEMP_SOL))

    DOMAIN = [(0, 9)] * (len(PEOPLE) * 2)
    # Random guesses at optimal solution
    RANDOM_SOL = random_opt(DOMAIN, sched_cost)
    print(sched_cost(RANDOM_SOL))
    print_sched(RANDOM_SOL)

    # Hill climbing approach
    HC_SOL = hill_climb_opt(DOMAIN, sched_cost)
    print(sched_cost(HC_SOL))
    print_sched(HC_SOL)

    # Annealing Approach
    ANNEAL_SOL = anneal_opt(DOMAIN, sched_cost)
    print(sched_cost(ANNEAL_SOL))
    print_sched(ANNEAL_SOL)

    # Genetic Approach
    GEN_SOL = genetic_opt(DOMAIN, sched_cost)
    print(sched_cost(GEN_SOL))
    print_sched(GEN_SOL)
    