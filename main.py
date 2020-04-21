# -*- coding: UTF-8 -*-

import argparse
from score_system import *

parser = argparse.ArgumentParser()

parser.add_argument('--cropped', default=False, help="答题卡是否裁切 (什么意思?)")
parser.add_argument('--model', type=str, default="model.ckpt", help="训练模型存储地点 (默认为model.ckpt)")
parser.add_argument('--train', default=False, help="是否先训练模型 (默认为False)")
parser.add_argument('--template', default="Template.json", help="识别模版JSON文件 (默认为Template.json)")
parser.add_argument('--output_dir', default="result_json", help="输出识别结果JSON文件地址 (默认为result_json/)")
parser.add_argument('--save_dir', default="cropped_img", help="输出裁切过的图片的地址 (默认为cropped_img/)")
parser.add_argument('--id_dir', default="hand_write_stu_num", help="输出手写学号的地址 (默认为hand_write_stu_num/)")
parser.add_argument('--dir', default="raw_img", help="学生原始答题卡地址 (默认为raw_img/)")
parser.add_argument('--datadir', default="dateset", help="训练数据地址 (默认为dataset/)")
parser.add_argument('--warm_start', default=False, help="Continue training model")
args = parser.parse_args()

def main():
    system = scoreSystem(args.template, args.dir, args.output_dir, args.isCrop, args.model, args.train, args.save_dir, args.id_dir, args.datadir, args.warm_start) 
    system.score()

if __name__ == '__main__':
    main()

