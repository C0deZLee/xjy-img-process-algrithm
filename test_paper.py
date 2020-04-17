# -*- coding: UTF-8 -*-

from MNIST import *
import os

class testPaper:
    def __init__(self, rawFileList, bucket, isCrop, id, template):
        self.result = {}
        self.pagesImage = [cv2.imread(os.path.join(bucket, x)) for x in rawFileList]
        self.rawFileList = rawFileList
        self.x0 = 147
        self.y0 = 136
        self.x1 = self.x0 + template["pages"][0]["Marker"]["x"] + template["pages"][0]["Marker"]["width"]
        self.y1 = self.y0 + template["pages"][0]["Marker"]["y"] + template["pages"][0]["Marker"]["height"]
        self.isCrop = isCrop
        self.id = id

    def crop(self, save_dir):
        filedir = os.path.join(save_dir, "student" + str(self.id))
        if (not(os.path.exists(save_dir))):
            os.mkdir(save_dir)
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir) 
        for i in range(len(self.pagesImage)):
            img = self.pagesImage[i][self.y0:self.y1, self.x0:self.x1]
            filename = "page" + str(i) + ".jpg"
            cv2.imwrite(os.path.join(filedir, filename), img)

    def cropHandwrittenQuestion(self, template, save_dir):
        idx = 0
        filedir = os.path.join(save_dir, "student" + str(self.id))
        if (not(os.path.exists(save_dir))):
            os.mkdir(save_dir)
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir) 
        for i in range(len(self.pagesImage)):
            for ques in template["pages"][i]["WriteQuestions"]:
                if ques["options"][0]["height"] > 200:
                    x = ques["options"][0]["x"]
                    y = ques["options"][0]["y"]
                    w = ques["options"][0]["width"]
                    h = ques["options"][0]["height"]
                    if (not(self.isCrop)):
                        x += self.x0
                        y += self.y0
                    img = self.pagesImage[i][y:y+h, x:x+w]
                    filename = "writequestion" + str(idx) + ".jpg"
                    cv2.imwrite(os.path.join(filedir, filename), img)
                    idx += 1


    def getTemplateId(self, template):
        self.result["templateId"] = template["templateId"]

    def getSingleRedMask(self, img):
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return mask
    
    def getAverageIntensityValue(self, pageIdx, x, y, width, height, isMask):
        if (not(self.isCrop)):
            x += self.x0
            y += self.y0
        img = self.pagesImage[pageIdx]
        roi = img[y:y+height, x:x+width]
        if (isMask):
            roi = self.getSingleRedMask(roi)
        roiMean = roi.sum() / (width * height * 3)
        return roiMean

    def identifyCode(self, template, model, id_dir):
        filedir = os.path.join(id_dir, "student" + str(self.id))
        if (not(os.path.exists(id_dir))):
            os.mkdir(id_dir)
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir) 
        self.result["studentCode"] = {}
        idString = ""
        x = template["pages"][0]["ID"]["x"]
        y = template["pages"][0]["ID"]["y"]
        w = template["pages"][0]["ID"]["width"]
        h = template["pages"][0]["ID"]["height"]
        if (not(self.isCrop)):
            x += self.x0
            y += self.y0
        xc = x + w / 2
        yc = y + h / 2
        xs = x
        xe = x + w
        ys = y
        ye = y + h
        for i in range(h):
            if (self.pagesImage[0].item(y+i, int(xc), 1) < 252):
                ys = y + i
                break
        for i in range(h):
            if (self.pagesImage[0].item(y+h-i, int(xc), 1) < 252):
                ye = y + h - i
                break
        for i in range(w):
            if (self.pagesImage[0].item(int(yc), x+i, 1) < 252):
                xs = x + i
                break
        for i in range(w):
            if (self.pagesImage[0].item(int(yc), x+w-i, 1) < 252):
                xe = x + w - i
                break    
        num = len(template["pages"][0]["ID"]) - 5 
        w = (xe - xs) / num
        h = ye - ys      
        for i in range(num):
            x = xs + i * w
            y = ys
            img = self.pagesImage[0][int(y)+7:int(y+h)-7, int(x)+5:int(x+w)-5]
            filename = "digit" + str(i) + ".jpg"
            cv2.imwrite(os.path.join(filedir, filename), img)
            digit = str(model.predict(img))
            idString += digit
        self.result["studentCode"]["text"] = ("handwritten ID identified: " + idString)

    def identifyCode2(self, template):
        self.result["studentCode2"] = {}
        idString = ""
        ID = template["pages"][0]["ID"]
        for i in range(len(ID) - 5):
            minAverage = 255
            digitId = "digit" + str(i)
            digit = ""
            rois = ID[digitId]
            for roi in rois:
                roiMean = self.getAverageIntensityValue(0, roi["x"], roi["y"], roi["width"], roi["height"], False)
                if (roiMean < minAverage):
                    minAverage = roiMean
                    if (minAverage > 200):
                        continue
                    digit = str(roi["Digit"])
            idString += digit
        self.result["studentCode2"]["text"] = ("filled ID identified: " + idString)

    def scoreSingleChoice(self, template, pageIdx, choiceIdx):
        choiceRes = {}
        choice = template["pages"][pageIdx]["Choice"][choiceIdx]
        choiceRes["TopicNumber"] = choice["SN"]
        choiceRes["Score"] = 0
        choiceRes["Mark"] = ""
        choiceRes["SingleMark"] = ""
        minAverage = 255
        for roi in choice["options"]:
            roiMean = self.getAverageIntensityValue(pageIdx, roi["x"], roi["y"], roi["width"], roi["height"], False)
            if (roiMean < minAverage):
                minAverage = roiMean
                if (minAverage > 200):
                    continue
                choiceRes["Score"] = roi["Answer"]
                choiceRes["SingleMark"] = roi["Mark"]
            if (roi["Answer"] != 0):
                choiceRes["Mark"] = roi["Mark"]
        return choiceRes

    def scoreChoices(self, template):
        pagesNum = len(template["pages"])
        self.result["Choice"] = []
        for i in range(pagesNum):
            choicesNum = len(template["pages"][i]["Choice"])
            for j in range(choicesNum):
                self.result["Choice"].append(self.scoreSingleChoice(template, i, j))
    
    def scoreSingleWriteQuestion(self, template, pageIdx, writeIdx):
        writeRes = {}
        write = template["pages"][pageIdx]["WriteQuestions"][writeIdx]
        writeRes["TopicNumber"] = write["SN"]
        writeRes["score"] = 0
        maxAverage = 0
        for roi in write["scores"]:
            roiMean = self.getAverageIntensityValue(pageIdx, roi["x"], roi["y"], roi["width"], roi["height"], True)
            if (roiMean > maxAverage):
                maxAverage = roiMean
                writeRes["score"] = roi["Score"]
        return writeRes
    
    def scoreWriteQuestions(self, template):
        pagesNum = len(template["pages"])
        self.result["WriteQuestion"] = []
        for i in range(pagesNum):
            writesNum = len(template["pages"][i]["WriteQuestions"])
            for j in range(writesNum):
                if template["pages"][i]["WriteQuestions"][j]["options"][0]["height"] > 200:
                    continue
                self.result["WriteQuestion"].append(self.scoreSingleWriteQuestion(template, i, j))

    def score(self, template, model, id_dir):
        self.getTemplateId(template)
        self.identifyCode(template, model, id_dir)
        self.identifyCode2(template)
        self.scoreChoices(template)
        self.scoreWriteQuestions(template)



