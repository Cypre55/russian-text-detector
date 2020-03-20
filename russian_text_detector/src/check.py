#!/usr/bin/env python3
# This Python file uses the following encoding: utf-8
import rospy
from russian_text_detector.msg import result
from russian_text_detector.msg import Data


def callback(data):
    mastertext = u'Родина'.encode('utf-8')
    print(mastertext)
    rospy.loginfo(data.text)
    pub = rospy.Publisher('/final_result', result, queue_size=10)
    rate = rospy.Rate(10)
    msg = result()
    msg.pos = data.pos
    if data.text == mastertext:
        msg.result = True
    else:
        msg.result = False
    print(msg.result)
    pub.publish(msg)
    rate.sleep()


if __name__ == '__main__':
    rospy.init_node('matching', anonymous=True)
    sub = rospy.Subscriber('/detection_result', Data, callback)
    rospy.spin()
