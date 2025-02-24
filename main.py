"""
200 OpenCV examples by youcans / OpenCV 例程 200 篇
Copyright: 2022, Shan Huang, youcans@qq.com
"""

# 【0606】基于投影变换实现图像矫正
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import cv2.cuda


def onMouseAction(event, x, y, flags, param):  # 鼠标交互 (左键选点右键完成)
    setpoint = (x, y)
    if event == cv.EVENT_LBUTTONDOWN:  # 鼠标左键点击
        pts.append(setpoint)  # 选中一个多边形顶点
        print("选择顶点 {}：{}".format(len(pts), setpoint))



if __name__ == '__main__':
    img = cv.imread("1-3.png")  # 读取彩色图像(BGR)
    imgCopy = img.copy()
    height, width = img.shape[:2]

    # 鼠标交互从输入图像选择 4 个顶点
    print("单击左键选择 4 个顶点 (左上-左下-右下-右上):")
    pts = []  # 初始化 ROI 顶点坐标集合
    status = True  # 开始绘图状态
    cv.namedWindow('origin')  # 创建图像显示窗口
    cv.setMouseCallback('origin', onMouseAction, status)  # 绑定回调函数
    while True:
        if len(pts) > 0:
            cv.circle(imgCopy, pts[-1], 5, (0, 0, 255), -1)  # 绘制最近一个顶点
        if len(pts) > 1:
            cv.line(imgCopy, pts[-1], pts[-2], (255, 0, 0), 2)  # 绘制最近一段线段
        if len(pts) == 4:  # 已有 4个顶点，结束绘制
            cv.line(imgCopy, pts[0], pts[-1], (255, 0, 0), 2)  # 绘制最后一段线段
            cv.imshow('origin', imgCopy)
            cv.waitKey(1000)
            break
        cv.imshow('origin', imgCopy)
        cv.waitKey(100)
    cv.destroyAllWindows()  # 释放图像窗口
    ptsSrc = np.array(pts)  # 列表转换为 (4,2) Numpy 数组
    # max_val = np.max(ptsSrc, axis=0)
    # min_val = np.min(ptsSrc, axis=0)
    # width = max_val[0] - min_val[0]
    # height = max_val[1] - min_val[1]
    # print("max_val:", max_val)
    # print("min_val:", min_val)
    print(ptsSrc)
    print(pts)
    # pts = [(460, 156), (127, 524), (549, 517), (671, 147)]
    # pts = [(341, 423),  # 左上
    #        (200, 485),  # 左下
    #        (506, 627),  # 右下
    #        (585, 484)]  # 右上
        # [341, 423],  # 左上
        # [200, 485],  # 坐下
        # [506, 627],  # 右下
        # [585, 484])  # 右上
    height = 100
    width = 200
    # imgCopy2 = img[249:1031,233:688]
    # cv.imwrite("4.png", imgCopy2)
    # 计算投影变换矩阵 MP
    ptsSrc = np.float32(pts)  # 列表转换为Numpy数组，图像 4 顶点坐标 (x,y)
    # x1, y1, x2, y2 = int(0.3*width), int(0.3*height), int(0.7*width), int(0.7*height)
    # ptsDst = np.float32([[x1,y1], [x1,y2], [x2,y2], [x2,y1]])  # 投影变换后 4 顶点坐标
    ptsDst = np.float32([[0, 0], [0, height], [width, height], [width, 0]])  # 投影变换后 4 顶点坐标
    MP = cv.getPerspectiveTransform(ptsSrc, ptsDst)

    # 投影变换
    dsize = (width, height)  # 输出图像尺寸 (w, h)
    # dsize = (650, 500)  # 输出图像尺寸 (w, h)
    perspect = cv.warpPerspective(img, MP, dsize, borderValue=(255, 255, 255))  # 投影变换
    print(img.shape, ptsSrc.shape, ptsDst.shape, MP.shape)

    # plt.figure(figsize=(9, 3.4))
    # plt.subplot(131), plt.axis('off'), plt.title("(1) Original")
    # plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # plt.subplot(132), plt.axis('off'), plt.title("(2) Selected vertex")
    # plt.imshow(cv.cvtColor(imgCopy, cv.COLOR_BGR2RGB))
    # plt.subplot(133), plt.axis('off'), plt.title("(3) Perspective correction")
    # plt.imshow(cv.cvtColor(perspect, cv.COLOR_BGR2RGB))
    cv.imwrite("Fig5_result.png", perspect)
    plt.tight_layout()
    plt.show()
