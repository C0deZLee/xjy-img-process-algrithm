# -*- coding: UTF-8 -*-

from MNIST import *

class testPaper:
    def __init__(self, rawFileList, bucket, isCrop):
        self.result = {}
        self.pagesImage = [cv2.imread(os.path.join(bucket, x)) for x in rawFileList]
        self.x0 = 147
        self.y0 = 136
        self.isCrop = isCrop

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

    def identifyCode(self, template, model):
        self.result["studentCode"] = {}
        idString = ""
        HandID = template["pages"][0]["HandID"]
        for i in range(len(HandID)):
            digitID = "digit" + str(i)
            roi = HandID[digitID]
            x = roi["x"]
            y = roi["y"]
            w = roi["width"]
            h = roi["height"]
            if (not(self.isCrop)):
                x += self.x0
                y += self.y0
            img = self.pagesImage[0][y:y+h, x:x+w]
            digit = str(model.predict(img))
            idString += digit
        self.result["studentCode"]["text"] = ("填涂学号识别结果：" + idString)

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
                    digit = str(roi["Digit"])
            idString += digit
        self.result["studentCode2"]["text"] = ("填涂学号识别结果：" + idString)

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
                if ("tenscores" in template["pages"][i]["WriteQuestions"][j]):
                    continue
                self.result["WriteQuestion"].append(self.scoreSingleWriteQuestion(template, i, j))

    def score(self, template, model):
        self.getTemplateId(template)
        self.identifyCode(template, model)
        self.identifyCode2(template)
        self.scoreChoices(template)
        self.scoreWriteQuestions(template)



