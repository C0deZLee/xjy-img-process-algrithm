# -*- coding: UTF-8 -*-

from test_paper import *
import json
import os

class scoreSystem:
    def __init__(self, template, raw_img_dir, result_json_dir, cropped, model_file, cropped_img_dir, hand_written_dir, data_dir, warm_start, bulk_load=True):
        # template                          识别模版JSON文件 (默认为Template.json)
        # raw_img                           学生原始答题卡地址 (默认为raw_img/)
        # result_json_dir                   识别结果储存地点 (默认为result_json/)
        # cropped                           答题卡是否裁切 (什么意思?)
        # model_file                        训练模型文件 (默认为model.ckpt)
        # cropped_img_dir                   输出裁切过的图片的地址 (默认为cropped_img/)
        # hand_written_dir                  输出手写学号的地址 (默认为hand_write_stu_num/)
        # data_dir                          训练数据地址 (默认为dataset/)
        # warm_start                        Continue training model
        # bulk_load                         是否一次性载入多张答题卡

        self.papers = [] # 答题卡列表
        self.cropped = cropped # If img is cropped
        self.template = template  # 识别模版JSON

        self.result_json_dir = result_json_dir    # 输出识别结果JSON文件地址
        self.cropped_img_dir = cropped_img_dir    # 输出裁切过的图片的地址
        self.hand_written_dir = hand_written_dir  # 输出手写学号的地址

        self.model = mnistModel(model_file, data_dir, warm_start)  # 识别模型
        self.train = train # 是否训练模型

        page_nums = len(self.template["pages"]) # 有几页
        
        if bulk_load:  # 读取文件夹下全部的学生原始答题卡
            file_list = []
            for files in os.walk(raw_img_dir):
                for file_ in files[2]:
                    file_list.append(file_)
            file_list.sort()
            file_nums = len(file_list)
            idx = 0
            while (idx < file_nums):
                cur = 0
                raw_file_list = []
                while (cur < page_nums):
                    if (idx >= file_nums):
                        break
                    if (file_list[idx].split('.')[-1] != 'jpg'):
                        idx += 1
                        continue
                    raw_file_list.append(file_list[idx])
                    cur += 1
                    idx += 1
                if (idx >= file_nums):
                    break
                
                raw_file_list = [os.path.join(raw_img_dir, raw_file_name) for raw_file_name in raw_file_list]
                self.papers.append(testPaper(raw_file_list, self.cropped, len(self.papers), self.template))
                print("Load test paper: " + str(len(self.papers)))

            else:
                pass
    
    def trainModel(self):
        if self.train:
            self.model.getdata()
            self.model.train()

    def score(self):
        idx = 0
        
        for paper in self.papers:
            if (not(self.cropped)):
                paper.crop(self.cropped_img_dir)
            paper.cropHandwrittenQuestion(self.cropped_img_dir)
            paper.score(self.model, self.hand_written_dir)
            dir = os.path.join(self.result_json_dir, "student" + str(idx) + ".json")
            if (not(os.path.exists(self.result_json_dir))):
                os.mkdir(self.result_json_dir)
            with open(dir, 'w') as f:
                f.write(json.dumps(paper.result))
            idx += 1
            print("Score finished: " + str(idx) + "/" + str(len(self.papers)))
            



