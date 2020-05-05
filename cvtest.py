import cv2
import numpy as np

import os

for (dirpath, dirnames, filenames) in os.walk("test_img 2"):
	for i in range(len(filenames)):

		filename = filenames[i]
		if (filename.split('.')[-1] != "jpg"):
			continue

		print(filename)
		img = cv2.imread(os.path.join(dirpath, filename))

		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (15, 15), 0)

		_, thresh = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV)

		cv2.imshow("", thresh)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		marker_w = 32
		marker_h = 32

		fcontours_top = []
		fcontours_bottom = []
		
		for contour in contours:
			x_max = 0
			y_max = 0
			x_min = thresh.shape[1]
			y_min = thresh.shape[0]
			for point in contour:
				x_max = max(x_max, point[0][0])
				x_min = min(x_min, point[0][0])
				y_max = max(y_max, point[0][1])
				y_min = min(y_min, point[0][1])
			if x_max - x_min >= marker_w - 5 and x_max - x_min <= marker_w + 5\
				 and y_max - y_min >= marker_h - 10 and y_max - y_min <= marker_h + 10:
				if y_min > 3150:
					# print(x_max - x_min, y_max - y_min)
					fcontour = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])
					fcontours_top.append(fcontour)
				elif y_max < 44:
					# print(x_max - x_min, y_max - y_min)
					fcontour = np.array([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])
					fcontours_bottom.append(fcontour)

		print(len(fcontours_top), len(fcontours_bottom))





