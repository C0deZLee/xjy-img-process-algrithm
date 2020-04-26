# -*- coding: UTF-8 -*-

from MNIST import *
import os
import zipfile

def findZero(img):
    x0 = img.shape[1]
    y0 = img.shape[0]
    for i in range(200):
        for j in range(200):
            if img.item(i, j, 1) < 100:
                x0 = min(x0, j)
                y0 = min(y0, i)
    return (x0, y0)

class testPaper:
    def __init__(self, rawFileList, bucket, isCrop, id, template):
        self.result = {"files": []}
        self.pagesImage = [cv2.imread(os.path.join(bucket, x)) for x in rawFileList]
        self.rawFileList = rawFileList
        for file_ in self.rawFileList:
            self.result["files"].append(file_)
        self.x0 = []
        self.y0 = []
        self.x1 = []
        self.y1 = []
        self.isCrop = isCrop
        self.template = template
        self.id = id

    def crop(self, save_dir):
        filedir = os.path.join(save_dir, "student" + str(self.id))
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir) 
        self.result["zipfile"] = []
        for i in range(len(self.pagesImage)):
            img = self.pagesImage[i]
            h = img.shape[0]
            w = img.shape[1]
            y0 = 0
            x0 = 0
            y1 = h - 1
            x1 = w - 1
            for j in range(h - 1, -1, -1):
                if img.item(j, int(w / 2), 1) > 250:
                    y1 = j
                    break
            for j in range(0, h):
                if img.item(j, int(w / 2), 1) > 250:
                    y0 = j
                    break
            for j in range(w - 1, -1, -1):
                if img.item(int(h / 2), j, 1) > 250:
                    x1 = j
                    break
            for j in range(0, w):
                if img.item(int(h / 2), j, 1) > 250:
                    x0 = j
                    break
            img = cv2.resize(img[y0:y1, x0:x1], (w, h), interpolation = cv2.INTER_CUBIC)
            (x0, y0) = findZero(img)
            self.x0.append(x0)
            self.y0.append(y0)
            self.x1.append(self.x0[-1] + self.template["pages"][0]["Marker"]["x"] + self.template["pages"][0]["Marker"]["width"])
            self.y1.append(self.y0[-1] + self.template["pages"][0]["Marker"]["y"] + self.template["pages"][0]["Marker"]["height"])
            img = img[self.y0[i]:self.y1[i], self.x0[i]:self.x1[i]]
            filename = "page" + str(i) + ".jpg"
            cv2.imwrite(os.path.join(filedir, filename), img)
            with zipfile.ZipFile(os.path.join(filedir, "page" + str(i) + ".zip"), 'w') as z:
                z.write(os.path.join(filedir, filename))
            self.result["zipfile"].append(os.path.join(filedir, "page" + str(i) + ".zip"))

    def cropName(self, id_dir):
        filedir = os.path.join(id_dir, "student" + str(self.id))
        x = self.template["pages"][0]["Name"]["x"]
        y = self.template["pages"][0]["Name"]["y"]
        w = self.template["pages"][0]["Name"]["width"]
        h = self.template["pages"][0]["Name"]["height"]
        if (not(self.isCrop)):
            x += self.x0[0]
            y += self.y0[0]
        filename = "name.jpg"
        img = self.pagesImage[0][y:y+h, x:x+w]
        cv2.imwrite(os.path.join(filedir, filename), img)
        self.result["studentName"] = {}
        self.result["studentName"]["path"] = os.path.join(filedir, filename)
        self.result["studentName"]["text"] = ""
        self.result["studentName"]["confidence"] = "0"

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
            x += self.x0[pageIdx]
            y += self.y0[pageIdx]
        img = self.pagesImage[pageIdx]
        roi = img[y:y+height, x:x+width]
        if (isMask):
            roi = self.getSingleRedMask(roi)
        roiMean = roi.sum() / (width * height * 3)
        return roiMean

    def identifyCode(self, model, id_dir):
        filedir = os.path.join(id_dir, "student" + str(self.id))
        if (not(os.path.exists(filedir))):
            os.mkdir(filedir) 
        self.result["studentCode"] = {}
        idString = ""
        x = self.template["pages"][0]["ID"]["x"]
        y = self.template["pages"][0]["ID"]["y"]
        w = self.template["pages"][0]["ID"]["width"]
        h = self.template["pages"][0]["ID"]["height"]
        if (not(self.isCrop)):
            x += self.x0[0]
            y += self.y0[0]
        filename = "id.jpg"
        img = self.pagesImage[0][y:y+h, x:x+w]
        cv2.imwrite(os.path.join(filedir, filename), img)
        self.result["studentCode"]["path"] = os.path.join(filedir, filename)
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
            x_tmp = xs + i * w
            y_tmp = ys
            img = 255 - self.pagesImage[0][int(y_tmp)+10:int(y_tmp+h)-10, int(x_tmp)+5:int(x_tmp+w)-5]
            filename = "digit" + str(i) + ".jpg"
            img2 = normalize(img)
            cv2.imwrite(os.path.join(filedir, filename), img2 * 255)
            prob, digit = model.predict(img)
            idString += str(digit)
            conf += str(prob)
        self.result["studentCode"]["text"] = ("handwritten ID identified: " + idString)
        self.result["studentCode"]["confidence"] = conf

    def identifyCode2(self):
        self.result["studentCode2"] = {"path": ""}
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
            x += self.x0[pageIdx]
            y += self.y0[pageIdx]
        img = self.pagesImage[pageIdx][y:y+h, x:x+w]
        if (write["options"][0]["JoinUp"] == 2):
            write2 = self.template["pages"][pageIdx + 1]["WriteQuestions"][0]
            x2 = write2["options"][0]["x"]
            y2 = write2["options"][0]["y"]
            w2 = write2["options"][0]["width"]
            h2 = write2["options"][0]["height"]
            if (not(self.isCrop)):
                x2 += self.x0[pageIdx + 1]
                y2 += self.y0[pageIdx + 1]
            img2 = self.pagesImage[pageIdx + 1][y2:y2+h2, x2:x2+w2]
            img = np.concatenate((img, img2))
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
        self.cropName(id_dir)
        self.identifyCode(model, id_dir)
        self.identifyCode2()
        self.scoreChoices()
        self.scoreWriteQuestions(save_dir)



