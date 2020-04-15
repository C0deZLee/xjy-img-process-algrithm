# -*- coding: UTF-8 -*-

import json
from score_system import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--isCrop', default=False, help="Whether the test paper is cropped.")
parser.add_argument('--model', type=str, default="model.ckpt", help="the file to save and load model.")
parser.add_argument('--train', default=False, help="Whether to train the model first or directly predict.")
args = parser.parse_args()

def main():
    with open("Template.json") as f:
        template = json.load(f)
    bucket = "Json Template with scanned pictures"
    outBucket = "Score Output"
    system = scoreSystem(template, bucket, outBucket, args.isCrop, args.model, args.train)   
    system.score()

if __name__ == '__main__':
    main()

