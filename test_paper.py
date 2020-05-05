import os
from .MNIST import *
import json

class testPaper:
    def __init__(self, rawFileFolder, rawFileNameList, template, modelPath=None, dataPath=None):
        """答题卡初始化"""
        # 答题卡原始文件路径列表
        self.rawFilePathList = [os.path.join(rawFileFolder, name) for name in rawFileNameList]
        # 答题卡的图片文件实例列表
        self.rawImages = [cv2.imread(f) for f in self.rawFilePathList] 
        # 答题卡识别模版
        self.template = template 
        # 答题卡ID
        self.id = rawFileNameList[0].split('.')[0]
        # 识别模型地址
        self.model = mnistModel(modelPath, dataPath, False)
        # 裁剪图片保存地址
        self.saveDir = os.path.join(rawFileFolder, self.id)
        if (not(os.path.exists(self.saveDir))):
            os.mkdir(self.saveDir)
        # 初始化返回的JSON文件
        self.resultJSON = {"files": self.rawFilePathList, 'templateId': self.template["templateId"]} 

    def cropName(self):
        """裁剪姓名区域"""
        fileName = self.id + "_student_name.jpg" # 文件名
        filePath = os.path.join(self.saveDir, fileName) # 文件路径

        # 获取裁切区域
        x = self.template["pages"][0]["Name"]["x"]
        y = self.template["pages"][0]["Name"]["y"]
        w = self.template["pages"][0]["Name"]["width"]
        h = self.template["pages"][0]["Name"]["height"]

        # 保存图片
        cv2.imwrite(filePath, self.rawImages[0][y:y+h, x:x+w])
        
        # 保存JSON
        self.resultJSON["studentName"] = {'path': filePath, 'text': '', 'confidence': ''}

    def cropStudentCode(self):
        """裁剪手写学号"""
        fileName = self.id + "_student_code1.jpg" # 文件名
        filePath = os.path.join(self.saveDir, fileName) # 文件路径

        # 获取裁切区
        x = self.template["pages"][0]["ID"]["x"]
        y = self.template["pages"][0]["ID"]["y"]
        w = self.template["pages"][0]["ID"]["width"]
        h = self.template["pages"][0]["ID"]["height"]

        # 保存图片
        cv2.imwrite(filePath, self.rawImages[0][y:y+h, x:x+w])

        # 保存JSON
        self.resultJSON["studentCode"] = {'path': filePath}

    def getSingleRedMask(self, img):
        """获取红线"""
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return mask

    def getAverageIntensityValue(self, pageIdx, x, y, width, height, isMask):
        img = self.rawImages[pageIdx]
        roi = img[y:y+height, x:x+width]
        if (isMask):
            roi = self.getSingleRedMask(roi)
        roiMean = roi.sum() / (width * height * 3)
        return roiMean

    def identifyCode(self):
        """识别手写学号"""

        # 获取裁切区
        x = self.template["pages"][0]["ID"]["x"]
        y = self.template["pages"][0]["ID"]["y"]
        w = self.template["pages"][0]["ID"]["width"]
        h = self.template["pages"][0]["ID"]["height"]

        xc = x + w / 2
        yc = y + h / 2
        xs = x
        xe = x + w
        ys = y
        ye = y + h
        for i in range(h):
            if (self.rawImages[0].item(y+i, int(xc), 1) < 220):
                ys = y + i
                break
        for i in range(h):
            if (self.rawImages[0].item(y+h-i, int(xc), 1) < 220):
                ye = y + h - i
                break
        for i in range(w):
            if (self.rawImages[0].item(int(yc), x+i, 1) < 220):
                xs = x + i
                break
        for i in range(w):
            if (self.rawImages[0].item(int(yc), x+w-i, 1) < 220):
                xe = x + w - i
                break
        num = 5
        w = (xe - xs) / num
        h = ye - ys
        conf = ""
        idString = ""
        if (not(os.path.exists("hand_written_code"))):
            os.mkdir("hand_written_code")
        for i in range(num):
            if i != 0:
                conf += ", "
            x_tmp = xs + i * w
            y_tmp = ys
            img = 255 - self.rawImages[0][int(y_tmp)+5:int(y_tmp+h)-5, int(x_tmp)+5:int(x_tmp+w)-5]
            filename = "digit" + str(i) + ".jpg"
            img = normalize(img[:, :, 0:1])
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))   
            # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
            # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)) 
            # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2, iterations=1)
            # img = cv2.erode(img, kernel) 
            cv2.imwrite(os.path.join("hand_written_code", filename), img[:, :, 0:1] * 255)
            img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
            # img = img[:, :, 0:1]
            img = np.expand_dims(img, 2)
            prob, digit = self.model.predict(img)
            idString += str(digit)
            conf += str(prob)
        self.resultJSON["studentCode"]["text"] = idString
        self.resultJSON["studentCode"]["confidence"] = conf
        print(idString, conf)

    def identifyCode2(self):
        """识别填涂学号"""
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
        self.resultJSON["studentCode2"] = {'text': idString}

    def scoreSingleChoice(self, pageIdx, choiceIdx):
        """识别单个选择题"""
        choiceRes = {}
        choice = self.template["pages"][pageIdx]["Choice"][choiceIdx]
        choiceRes["TopicNumber"] = choice["SN"]
        choiceRes["Score"] = 0
        choiceRes["Mark"] = ""
        choiceRes["SingleMark"] = ""
        minAverage = 255
        for roi in choice["options"]:
            roiMean = self.getAverageIntensityValue(pageIdx, roi["x"], roi["y"], roi["width"], roi["height"], False)
            if (roiMean <= 150):
                if (choiceRes["Mark"] != ""):
                    choiceRes["Mark"] += ", "
                choiceRes["Mark"] += roi["Mark"]
            if (roiMean < minAverage):
                minAverage = roiMean
                if (minAverage > 150):
                    continue
                choiceRes["Score"] = roi["Answer"]
                choiceRes["SingleMark"] = roi["Mark"]
        return choiceRes

    def scoreChoices(self):
        """识别选择题"""
        pagesNum = len(self.template["pages"])
        self.resultJSON["Choice"] = []
        for i in range(pagesNum):
            choicesNum = len(self.template["pages"][i]["Choice"])
            for j in range(choicesNum):
                self.resultJSON["Choice"].append(self.scoreSingleChoice(i, j))

    def scoreSingleWriteQuestion(self, pageIdx, writeIdx):
        """识别单个主观题"""
        writeRes = {}
        write = self.template["pages"][pageIdx]["WriteQuestions"][writeIdx]
        writeRes["TopicNumber"] = write["SN"]
        writeRes["score"] = 0
        maxAverage = 0
        for roi in write["scores"]:
            roiMean = self.getAverageIntensityValue(pageIdx, int(roi["x"]), int(
                roi["y"]), int(roi["width"]), int(roi["height"]), True)
            if (roiMean > maxAverage):
                maxAverage = roiMean
                writeRes["score"] = roi["Score"]
        if ("tenscores" in write):
            tenscore = 0
            maxAverage = 0
            for roi in write["tenscores"]:
                roiMean = self.getAverageIntensityValue(pageIdx, int(roi["x"]), int(
                    roi["y"]), int(roi["width"]), int(roi["height"]), True)
                if (roiMean > maxAverage):
                    maxAverage = roiMean
                    if (maxAverage > 2):
                        tenscore = roi["Score"]
            writeRes["score"] += tenscore

        # 裁切主观题图片
        fileName = self.id + "_writequestion_" + str(write["SN"]) + ".jpg" # 文件名
        filePath = os.path.join(self.saveDir, fileName) # 文件路径

        # 获取裁切区域
        crossPage = False
        x = write["options"][0]["x"]
        y = write["options"][0]["y"]
        w = write["options"][0]["width"]
        h = write["options"][0]["height"]
        img = self.rawImages[pageIdx][y:y+h, x:x+w]
        
        # 跨页合并
        if (write["options"][0]["JoinUp"] == 2):
            crossPage = True
            write2 = self.template["pages"][pageIdx + 1]["WriteQuestions"][0]
            x2 = write2["options"][0]["x"]
            y2 = write2["options"][0]["y"]
            w2 = write2["options"][0]["width"]
            h2 = write2["options"][0]["height"]
            img2 = self.rawImages[pageIdx + 1][y2:y2+h2, x2:x2+w2]
            img = np.concatenate((img, img2))

        # 保存图片
        cv2.imwrite(filePath, img)
        
        # 保存JSON
        writeRes["Items"] = [{"Note": "跨页" if crossPage else "", "ItemID": 1, "path": filePath}]
        return writeRes

    def scoreWriteQuestions(self):
        """识别主观题"""
        pagesNum = len(self.template["pages"])
        self.resultJSON["WriteQuestion"] = []
        for i in range(pagesNum):
            writesNum = len(self.template["pages"][i]["WriteQuestions"])
            for j in range(writesNum):
                self.resultJSON["WriteQuestion"].append(self.scoreSingleWriteQuestion(i, j))

    def score(self):
        # 裁切姓名
        self.cropName()
        # 裁切手写学号
        self.cropStudentCode()
        # 识别手写学号
        self.identifyCode()
        # 识别填涂学号
        self.identifyCode2()
        # 识别选择题
        self.scoreChoices()
        # 识别主观题
        self.scoreWriteQuestions()
        
        # 返回 result JSON
        return self.resultJSON

# with open("Template2.json") as f:
#     template = json.load(f)

# test_paper = testPaper(".", ["E77673E9X112747_20191028_090252_0002.min.jpg", "E77673E9X112747_20191028_090252_0001.min.jpg"], template, "model.ckpt")
# test_paper.identifyCode()