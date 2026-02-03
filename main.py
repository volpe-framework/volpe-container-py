import struct
import threading
import concurrent.futures
from array import array
from typing import override

import grpc
import tsplib95 as tsplib
from deap import base, creator, tools, algorithms
import random

# Protobuf imports
import volpe_container_pb2 as pb
import common_pb2 as pbc
import volpe_container_pb2_grpc as vp

# --- TSP Setup ---
problem = tsplib.load_problem('gil262.tsp')
NDIM = 262
BASE_POPULATION_SIZE = 100
LAMBDA_SIZE = BASE_POPULATION_SIZE*7
CXPROB = 0.5
MUTATION_RATE = 0.2

# --- DEAP Configuration ---
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# We define our Individual as a subclass of array.array
creator.create("Individual", array, fitness=creator.FitnessMin, typecode='i')

toolbox = base.Toolbox()

toolbox.register('indices', random.select, range(1, NDIM+1))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_tsp(individual):
    # Convert array back to list for tsplib if necessary
    return (problem.trace_tours([list(individual)])[0],)

toolbox.register("evaluate", evaluate_tsp)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# --- Serialization Utilities ---
# 'i' is signed 4-byte integer.
STRUCT_FORMAT = f">{NDIM}i"

def ind_to_bytes(individual: array) -> bytes:
    """Converts array-based individual to bytes."""
    return struct.pack(STRUCT_FORMAT, *individual)

def bytes_to_ind(data: bytes):
    """Converts bytes back to a DEAP array Individual."""
    indices = struct.unpack(STRUCT_FORMAT, data)
    return creator.Individual('i', indices)

# --- Servicer Implementation ---
class VolpeGreeterServicer(vp.VolpeContainerServicer):
    def __init__(self):
        self.popln = toolbox.population(n=BASE_POPULATION_SIZE)
        # Evaluate initial population
        self.popln = toolbox.population(n=BASE_POPULATION_SIZE)
        for ind in self.popln:
            ind.fitness.values = toolbox.evaluate(ind)
        self.poplock = threading.Lock()
        self.last_best = float('inf')

    @override
    def InitFromSeed(self, request, context):
        with self.poplock:
            self.popln = toolbox.population(n=BASE_POPULATION_SIZE)
            for ind in self.popln:
                ind.fitness.values = toolbox.evaluate(ind)
            return pb.Reply(success=True)

    @override
    def InitFromSeedPopulation(self, request: pbc.Population, context):
        with self.poplock:
            indices = random.sample(range(len(self.popln)), len(request.members))
            for memb, idx in zip(request.members, indices):
                new_ind = bytes_to_ind(memb.genotype)
                new_ind.fitness.values = (memb.fitness,)
                self.popln[idx] = new_ind
            return pb.Reply(success=True)

    @override
    def GetBestPopulation(self, request, context):
        with self.poplock:
            best_inds = tools.selBest(self.popln, request.size)
            members = [
                pbc.Individual(
                    genotype=ind_to_bytes(ind), 
                    fitness=ind.fitness.values[0]
                ) for ind in best_inds
            ]
            return pbc.Population(members=members)

    @override
    def RunForGenerations(self, request, context):
        with self.poplock:
            self.popln, logbook = algorithms.eaMuPlusLambda(
                population=self.popln,
                toolbox=toolbox,
                mu=BASE_POPULATION_SIZE,
                lambda_= LAMBDA_SIZE,
                cxpb=CXPROB,
                mutpb=MUTATION_RATE,
                ngen=1,        # We run 1 generation per RPC call
                verbose=False
            )
            current_best = tools.selBest(self.popln, 1)[0].fitness.values[0]
            self.last_best = current_best
            
            return pb.Reply(success=True)

if __name__ == '__main__':
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    vp.add_VolpeContainerServicer_to_server(VolpeGreeterServicer(), server)
    server.add_insecure_port("0.0.0.0:8081")
    server.start()
    server.wait_for_termination()
