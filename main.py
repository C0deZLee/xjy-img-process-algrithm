# -*- coding: UTF-8 -*-

import json
from score_system import *

def main():
    with open("Template.json") as f:
        template = json.load(f)
    bucket = "Json Template with scanned pictures"
    outBucket = "Score Output"
    system = scoreSystem(template, bucket, outBucket, False)   
    system.score()

if __name__ == '__main__':
    main()

