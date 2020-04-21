# -*- coding: UTF-8 -*-

from MNIST import *
import os

class testPaper:
    def __init__(self, rawFileNameList, rawFileDir, isCrop, id, template):
        self.result = {}
        self.pagesImage = [cv2.imread(os.path.join(rawFileDir, fileName)) for fileName in rawFileNameList]
        self.rawFileDir = rawFileDir
        self.rawFileNameList = rawFileNameList
        self.x0 = 147
        self.y0 = 136
        self.x1 = self.x0 + template["pages"][0]["Marker"]["x"] + template["pages"][0]["Marker"]["width"]
        self.y1 = self.y0 + template["pages"][0]["Marker"]["y"] + template["pages"][0]["Marker"]["height"]
        self.isCrop = isCrop
        self.template = template
        self.id = id

    def crop(self, save_dir):
        """裁剪答题卡并返回本地路径"""
        filedir = os.path.join(self.rawFileDir, "cropped_sheets")

        # 创建文件路径
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir)

        croppedSheetList = []

        for i in range(len(self.pagesImage)):
            #TODO: 如果遇到扭曲的图片需要额外处理
            img = self.pagesImage[i][self.y0:self.y1, self.x0:self.x1]


            # 将裁切后的答题卡存储下来
            filename = self.rawFileNameList[i].replace('.jpg', '') + '_cropped_' + str(i) + ".jpg"
            filepath = os.path.join(filedir, filename)
            cv2.imwrite(filepath, img)
            croppedSheetList.append(filepath)

        return croppedSheetList
        
    def cropHandwrittenQuestion(self):
        """裁剪手写题并返回手写题的本地路径"""
        filedir = os.path.join(self.rawFileDir, "cropped_write_questions")
    
        # 创建文件路径
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir)
        
        # 返回的文件名列表
        croppedWriteQuestionsList = []
        
        # 遍历答题卡中的手写题
        idx = 0
        for i in range(len(self.pagesImage)):
            for ques in self.template["pages"][i]["WriteQuestions"]:
                if ques["options"][0]["height"] > 200:
                    x = ques["options"][0]["x"]
                    y = ques["options"][0]["y"]
                    w = ques["options"][0]["width"]
                    h = ques["options"][0]["height"]
                    if (not(self.isCrop)):
                        x += self.x0
                        y += self.y0
                    img = self.pagesImage[i][y:y+h, x:x+w]
                    filename = self.rawFileNameList[i].replace('.jpg', '') + '_write_question_' + str(idx) + ".jpg"
                    
                    # 将裁切后的答题卡存储下来
                    filepath = os.path.join(filedir, filename)
                    cv2.imwrite(filepath, img)
                    croppedWriteQuestionsList.append(filepath)
                    
                    idx += 1
        
        return croppedWriteQuestionsList

    def getTemplateId(self):
        self.result["templateId"] = self.template["templateId"]

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

    def identifyCode(self, model, id_dir):
        filedir = os.path.join(id_dir, "student" + str(self.id))
        if (not(os.path.exists(id_dir))):
            os.mkdir(id_dir)
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir) 
        self.result["studentCode"] = {}
        idString = ""
        x = self.template["pages"][0]["ID"]["x"]
        y = self.template["pages"][0]["ID"]["y"]
        w = self.template["pages"][0]["ID"]["width"]
        h = self.template["pages"][0]["ID"]["height"]
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
        num = len(self.template["pages"][0]["ID"]) - 5 
        w = (xe - xs) / num
        h = ye - ys    
        conf = ""
        for i in range(num):
            if i != 0:
                conf += ", "
            x = xs + i * w
            y = ys
            img = 255 - self.pagesImage[0][int(y)+10:int(y+h)-10, int(x)+5:int(x+w)-5]
            filename = "digit" + str(i) + ".jpg"
            img2 = np.where(normalize(img) > 0.3, 1, 0)
            img2 = np.concatenate((img2, img2, img2), 2)
            kernel = np.ones((3, 3), np.uint8)
            img2 = cv2.morphologyEx(np.uint8(img2), cv2.MORPH_CLOSE, kernel)
            kernel = np.ones((4, 4), np.uint8)
            img2 = cv2.morphologyEx(np.uint8(img2), cv2.MORPH_OPEN, kernel)
            kernel = np.ones((2, 2), np.uint8)
            img2 = cv2.erode(np.uint8(img2), kernel)
            cv2.blur(img2, (5, 5))
            cv2.imwrite(os.path.join(filedir, filename), img2 * 255)
            prob, digit = model.predict(img)
            idString += str(digit)
            conf += str(prob)
        self.result["studentCode"]["text"] = ("handwritten ID identified: " + idString)
        self.result["studentCode"]["confidence"] = conf

    def identifyCode2(self):
        self.result["studentCode2"] = {}
        idString = ""
        ID = self.template["pages"][0]["ID"]
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

    def scoreSingleChoice(self, pageIdx, choiceIdx):
        choiceRes = {}
        choice = self.template["pages"][pageIdx]["Choice"][choiceIdx]
        choiceRes["TopicNumber"] = choice["SN"]
        choiceRes["Score"] = 0
        choiceRes["Mark"] = ""
        choiceRes["SingleMark"] = ""
        minAverage = 255
        for roi in choice["options"]:
            roiMean = self.getAverageIntensityValue(pageIdx, roi["x"], roi["y"], roi["width"], roi["height"], False)
            if (roiMean < minAverage):
                minAverage = roiMean
                if (minAverage > 240):
                    continue
                choiceRes["Score"] = roi["Answer"]
                choiceRes["SingleMark"] = roi["Mark"]
            if (roi["Answer"] != 0):
                choiceRes["Mark"] = roi["Mark"]
        return choiceRes

    def scoreChoices(self):
        pagesNum = len(self.template["pages"])
        self.result["Choice"] = []
        for i in range(pagesNum):
            choicesNum = len(self.template["pages"][i]["Choice"])
            for j in range(choicesNum):
                self.result["Choice"].append(self.scoreSingleChoice(i, j))
    
    def scoreSingleWriteQuestion(self, pageIdx, writeIdx, save_dir):
        writeRes = {}
        write = self.template["pages"][pageIdx]["WriteQuestions"][writeIdx]
        writeRes["TopicNumber"] = write["SN"]
        writeRes["score"] = 0
        maxAverage = 0
        for roi in write["scores"]:
            roiMean = self.getAverageIntensityValue(pageIdx, int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"]), True)
            if (roiMean > maxAverage):
                maxAverage = roiMean
                writeRes["score"] = roi["Score"]
        if ("tenscores" in write):
            tenscore = 0
            maxAverage = 0
            for roi in write["tenscores"]:
                roiMean = self.getAverageIntensityValue(pageIdx, int(roi["x"]), int(roi["y"]), int(roi["width"]), int(roi["height"]), True)
                if (roiMean > maxAverage):
                    maxAverage = roiMean
                    if (maxAverage > 2):
                        tenscore = roi["Score"]
            writeRes["score"] += tenscore
        x = write["options"][0]["x"]
        y = write["options"][0]["y"]
        w = write["options"][0]["width"]
        h = write["options"][0]["height"]
        if (not(self.isCrop)):
            x += self.x0
            y += self.y0
        img = self.pagesImage[pageIdx][y:y+h, x:x+w]
        if (write["options"][0]["JoinUp"] == 2):
            write2 = self.template["pages"][pageIdx + 1]["WriteQuestions"][0]
            x2 = write2["options"][0]["x"]
            y2 = write2["options"][0]["y"]
            w2 = write2["options"][0]["width"]
            h2 = write2["options"][0]["height"]
            if (not(self.isCrop)):
                x2 += self.x0
                y2 += self.y0
            img2 = self.pagesImage[pageIdx + 1][y2:y2+h2, x2:x2+w2]
            img = np.concatenate((img, img2))
        if (not(os.path.exists(save_dir))):
            os.mkdir(save_dir)
        filename = "writequestion_" + str(write["SN"]) + ".jpg"
        filedir = os.path.join(save_dir, "student" + str(self.id))
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir)
        cv2.imwrite(os.path.join(filedir, filename), img)
        writeRes["Items"] = [{"Note": "Only one block", "ItemID": 1, "path": os.path.join(filedir, filename)}]
        return writeRes
    
    def scoreWriteQuestions(self, save_dir):
        pagesNum = len(self.template["pages"])
        self.result["WriteQuestion"] = []
        for i in range(pagesNum):
            writesNum = len(self.template["pages"][i]["WriteQuestions"])
            for j in range(writesNum):
                self.result["WriteQuestion"].append(self.scoreSingleWriteQuestion(i, j, save_dir))

    def score(self, model, id_dir, save_dir):
        self.getTemplateId()
        self.identifyCode(model, id_dir)
        self.identifyCode2()
        self.scoreChoices()
        self.scoreWriteQuestions(save_dir)



