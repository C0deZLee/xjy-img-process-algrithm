# -*- coding: UTF-8 -*-

from test_paper import *
import json

class scoreSystem:
    def __init__(self, template, bucket, outBucket, isCrop, reTrain, filename, trainAgain):
        self.papers = []
        self.isCrop = isCrop
        self.template = template
        self.outBucket = outBucket
        self.model = mnistModel(reTrain, filename)
        self.trainAgain = trainAgain
        pageNums = len(self.template["pages"])
        filelist = []
        for files in os.walk(bucket):
            for file_ in files[2]:
                filelist.append(file_)
        filelist.sort()
        fileNums = len(filelist)
        for file_ in filelist:
            print(file_)
        idx = 0
        while (idx < fileNums):
            cur = 0
            rawFileList = []
            while (cur < pageNums):
                if (idx >= fileNums):
                    break
                if (filelist[idx].split('.')[-1] != 'jpg'):
                    idx += 1
                    continue
                rawFileList.append(filelist[idx])
                cur += 1
                idx += 1
            if (idx >= fileNums):
                break
            self.papers.append(testPaper(rawFileList, bucket, self.isCrop))
            print("Load test paper: " + str(len(self.papers)))
    
    def score(self):
        idx = 0
        if (self.trainAgain):
            self.model.train()
        for paper in self.papers:
            paper.score(self.template, self.model)
            dir = os.path.join(self.outBucket, "student" + str(idx) + ".json")
            if (not(os.path.exists(self.outBucket))):
                os.mkdir(self.outBucket)
            with open(dir, 'w') as f:
                f.write(json.dumps(paper.result))
            idx += 1
            print("Score finished: " + str(idx) + "/" + str(len(self.papers)))
            



