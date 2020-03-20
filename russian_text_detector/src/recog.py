#!/usr/bin/env python3
# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from russian_text_detector.msg import Data

br = CvBridge()


def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)


def callback(data):
    rospy.loginfo("")
    image = br.imgmsg_to_cv2(data)
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    (newW, newH) = (320, 320)
    rW = origW / float(newW)
    rH = origH / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(
        "/home/chaoticsaint/Desktop/ARK/catkin_ws/src/russian_text_detector/src/frozen_east_text_detection.pb")

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    results = []

    for (startX, startY, endX, endY) in boxes:
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(origW, endX)
        endY = min(origH, endY)

        roi = orig[startY:endY, startX:endX]

        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        results.append(((startX, startY, endX, endY), text))

    results = sorted(results, key=lambda r: r[0][1])
    pub = rospy.Publisher('/detection_result', Data, queue_size=10)
    rate = rospy.Rate(10)

    for ((startX, startY, endX, endY), text) in results:
        msg = Data()
        msg.text = text
        msg.pos.x = float(startX + endX) / 2
        msg.pos.y = float(startY + endY) / 2
        msg.pos.z = 0.0

        pub.publish(msg)
        rospy.loginfo("MSG Published...")
        rate.sleep()


if __name__ == '__main__':
    rospy.init_node('detection', anonymous=True)
    sub = rospy.Subscriber('/videofeed', Image, callback)
    rospy.spin()
