#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

br = CvBridge()


def camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)
    pub = rospy.Publisher('/videofeed', Image, queue_size=10)
    rospy.init_node('camera', anonymous=True)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        _, frame = cap.read()
        rospy.loginfo("videofeed published...")
        pub.publish(br.cv2_to_imgmsg(frame))
        rate.sleep()


if __name__ == '__main__':
    try:
        camera()
    except rospy.ROSInterruptException:
        pass
