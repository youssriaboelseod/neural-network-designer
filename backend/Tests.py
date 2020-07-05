from backend.nnb import *
from backend.tools import *
import argparse

def main():
    neurs, acts = get_random_model_scheme()
    nnb = NeuralNetworkDesigner(neurs, acts, str((-50, 50, 25)), 'x**3-x**2+5*x')
    arp = argparse.ArgumentParser()
    arp.add_argument('expression', type=str)
    arp.add_argument('--linspace', type=str)
    arp.add_argument('-plot', action='store_true')
    p = arp.parse_args()
    res = nnb.simulated_annealing(35, p)


main()
