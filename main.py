# -*- coding: UTF-8 -*-

import json
from score_system import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cropped', default=False, help="答题卡是否裁切 (什么意思?)")
parser.add_argument('--model', type=str, default="model.ckpt", help="训练模型存储地点 (默认为model.ckpt)")
parser.add_argument('--train', default=False, help="是否先训练模型 (默认为False)")
parser.add_argument('--template', default="Template.json", help="识别模版JSON文件 (默认为Template.json)")
parser.add_argument('--output_dir', default="result_json", help="输出识别结果JSON文件地址 (默认为result_json/)")
parser.add_argument('--save_dir', default="cropped_img", help="输出裁切过的图片的地址 (默认为cropped_img/)")
parser.add_argument('--id_dir', default="hand_write_stu_num", help="输出手写学号的地址 (默认为hand_write_stu_num/)")
parser.add_argument('--dir', default="raw_img", help="学生原始答题卡地址 (默认为raw_img/)")
args = parser.parse_args()

def main():
    with open(args.template) as f:
        template = json.load(f)
    
    system = scoreSystem(template, args.dir, args.output_dir, args.cropped, args.model, args.train, args.save_dir, args.id_dir)   
    system.score()

if __name__ == '__main__':
    main()

