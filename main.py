# -*- coding: UTF-8 -*-

import json
from score_system import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--isCrop', default=False, help="Whether the test paper is cropped.")
parser.add_argument('--model', type=str, default="model.ckpt", help="the file to save and load model.")
parser.add_argument('--train', default=False, help="Whether to train the model first or directly predict.")
parser.add_argument('--template', default="Template.json", help="The file of the template")
parser.add_argument('--output_dir', default="Score Output", help="The dir to save output result.json")
parser.add_argument('--save_dir', default="Cleaned image", help="The directory for the cropped paper")
parser.add_argument('--id_dir', default="Handwritten ID output")
parser.add_argument('--dir', default="Json Template with scanned pictures", help="The dir of the students' answers")
args = parser.parse_args()

def main():
    with open(args.template) as f:
        template = json.load(f)
    system = scoreSystem(template, args.dir, args.output_dir, args.isCrop, args.model, args.train, args.save_dir, args.id_dir)   
    system.score()

if __name__ == '__main__':
    main()

