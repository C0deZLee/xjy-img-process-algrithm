
import json
import os

from test_paper import *

class scoreSystem:
    def __init__(self, template, raw_img_dir, cropped, model_file, data_dir, warm_start, bulk_load, raw_file_path_list):
        """
        @Params
        template                          识别模版JSON文件 (默认为Template.json)
        raw_img                           学生原始答题卡地址 (默认为raw_img/)
        cropped                           答题卡是否裁切 (什么意思?)
        model_file                        训练模型文件 (默认为model.ckpt)
        data_dir                          训练数据地址 (默认为dataset/)
        warm_start                        Continue training model
        bulk_load                         是否一次性载入多张答题卡
        raw_file_list                     学生原始答题列表,以逗号隔开,最多四个
        """
        self.papers = [] # 答题卡列表
        self.cropped = cropped # If img is cropped
        self.template = template  # 识别模版JSON

        self.raw_img_dir = raw_img_dir # 原始文件地址

        self.model = mnistModel(model_file, data_dir, warm_start)  # 识别模型

        self.bulk_load = bulk_load  # 是否一次性载入多张答题卡

        page_nums = len(self.template["pages"]) # 有几页

        if bulk_load:  # 读取文件夹下全部的学生原始答题卡
            file_list = []

            for (dirpath, dirnames, filenames) in os.walk(raw_img_dir):
                file_list.extend(filenames)
                break
            
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
                
                self.papers.append(testPaper(raw_file_list, self.raw_img_dir, self.cropped,
                                             raw_file_list[0].replace('.jpg', ''), self.template))
                print("Load test paper: " + str(len(self.papers)))

        else:
            raw_file_list = raw_file_path_list.split(',')
            print(raw_file_list)
            self.papers.append(testPaper(raw_file_list, self.raw_img_dir,
                                            self.cropped, raw_file_list[0].replace('.jpg', ''), self.template))
            print("Load test paper: " + str(len(self.papers)))

    
    def trainModel(self):
        """训练模型"""
        self.model.getdata()
        self.model.train()

    def score(self):
        """判分"""
        if (not(os.path.exists(self.raw_img_dir))):
            os.mkdir(self.raw_img_dir)

        cropped_sheets_dir = os.path.join(self.raw_img_dir, "cropped_sheets")

        if (not(os.path.exists(cropped_sheets_dir))):
            os.mkdir(cropped_sheets_dir)
        hand_written_student_code_dir = os.path.join(self.raw_img_dir, "hand_written_student_code")

        if (not(os.path.exists(hand_written_student_code_dir))):
            os.mkdir(hand_written_student_code_dir)

        cropped_write_questions_dir = os.path.join(self.raw_img_dir, "cropped_write_questions")

        if (not(os.path.exists(cropped_write_questions_dir))):
            os.mkdir(cropped_write_questions_dir)

        result_json_dir = os.path.join(self.raw_img_dir, "result_json")

        if (not(os.path.exists(result_json_dir))):
            os.mkdir(result_json_dir)

        if self.bulk_load:  # 读取文件夹下全部的学生原始答题卡

            for idx, paper in enumerate(self.papers):

                # 如果不是裁剪过的图片, 则裁剪图片
                if not self.cropped:
                    cropped_list = paper.crop(cropped_sheets_dir)

                paper.score(self.model, hand_written_student_code_dir, cropped_write_questions_dir)
                
                # 创建文件路径
                filedir = result_json_dir

                # 保存result_json
                filepath = os.path.join(filedir, "result_json" + str(paper.id) + ".json")
                
                with open(filepath, 'w') as f:
                    f.write(json.dumps(paper.result))

                print("Score finished: " + str(paper.id) + "  " + str(idx+1) + "/" + str(len(self.papers)))
        else:
            paper = self.papers[0]
            idx = paper.id
            # 如果不是裁剪过的图片, 则裁剪图片
            if not self.cropped:
                cropped_list = paper.crop(cropped_sheets_dir)

            paper.score(self.model, hand_written_student_code_dir, cropped_write_questions_dir)

            # 创建文件路径
            filedir = result_json_dir

            # 保存result_json
            filepath = os.path.join(filedir, "result_json" + str(idx) + ".json")

            with open(filepath, 'w') as f:
                f.write(json.dumps(paper.result))

            print("Score finished: " + str(paper.id))

            return (filepath, paper.result)
