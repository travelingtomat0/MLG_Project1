import numpy as np
import argparse
import csv

# Use argument parser so that we call the main file
parser = argparse.ArgumentParser(description='Gene Expression Prediction Model.')
parser.add_argument('path', type=str, default='./',
                    help='path to the model data.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for low computational impact.')
args = parser.parse_args()

if __name__ == '__main__':
    print("Hello World")