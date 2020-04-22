# -*- coding: UTF-8 -*-

import argparse
from score_system import *

parser = argparse.ArgumentParser()

parser.add_argument('--cropped', default=False, help="答题卡是否裁切 (什么意思?)")
parser.add_argument('--model', type=str, default="model.ckpt", help="训练模型存储地点 (默认为model.ckpt)")
parser.add_argument('--train', default=False, help="是否先训练模型 (默认为False)")
parser.add_argument('--template', default="Template.json", help="识别模版JSON文件 (默认为test_data/template.json)")
parser.add_argument('--dir', default="raw_img", help="学生原始答题卡地址 (test_data/)")
parser.add_argument('--data_dir', default="dateset", help="训练数据地址 (默认为dataset/)")
parser.add_argument('--warm_start', default=False, help="Continue training model")
parser.add_argument('--bulk_load', default=False, help="是否从文件夹批量载入")
parser.add_argument(
    '--raw_file_list', default="E77673B9X114047_12102019_084354_0017.jpg,E77673B9X114047_12102019_084354_0018.jpg", help="学生原始答题列表,以逗号隔开,最多四个")
args = parser.parse_args()


def main():
    with open(args.template) as f:
        template = json.load(f)

    system = scoreSystem(template, args.dir,  args.cropped, args.model,
                         args.data_dir, args.warm_start, args.bulk_load, args.raw_file_list)
    if args.train:
        system.trainModel()

    system.score()


if __name__ == '__main__':
    main()
