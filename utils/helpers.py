import cv2 as cv
import numpy as np


'''
OpenCV实现
输入  bgr格式图片
输出  列表，每个都是框
'''
def search_search(image):
    ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    return rects


def calculate_iou(bbox1, bbox2):
    x_left = max(bbox1['x1'], bbox2['x1'])
    y_top = max(bbox1['y1'], bbox2['y1'])
    x_right = min(bbox1['x2'], bbox2['x2'])
    y_bottom = min(bbox1['y2'], bbox2['y2'])
    # 若二者不重叠，返回0
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    # 计算重叠部分
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    bb1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
    bb2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
    # 计算并集
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area

