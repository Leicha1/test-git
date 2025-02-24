import cv2
import numpy as np
from matplotlib import pyplot as plt
import time


def test_warp(img):
    pts = [(472, 214),  #左上
           (590, 217),  #右上
           (578, 478),  #右下
           (264, 463)]  #左下

    height = 850
    width = 400
    # 计算投影变换矩阵 MP
    ptsSrc = np.float32(pts)  # 列表转换为Numpy数组，图像 4 顶点坐标 (x,y)
    # x1, y1, x2, y2 = int(0.3*width), int(0.3*height), int(0.7*width), int(0.7*height)
    # ptsDst = np.float32([[x1,y1], [x1,y2], [x2,y2], [x2,y1]])  # 投影变换后 4 顶点坐标
    ptsDst = np.float32([
            [0, 0],             # 左上
            [400, 0],        # 右上
            [400, 850],    # 右下
            [0, 850]])        # 左下        # 投影变换后 4 顶点坐标
    MP = cv2.getPerspectiveTransform(ptsSrc, ptsDst)

    # 投影变换
    dsize = (width, height)  # 输出图像尺寸 (w, h)
    # dsize = (650, 500)  # 输出图像尺寸 (w, h)
    perspect = cv2.warpPerspective(img, MP, dsize, borderValue=(255, 255, 255))  # 投影变换
    # cv2.imshow("1", perspect)
    # cv2.waitKey(0)
    return perspect
    # print(img.shape, ptsSrc.shape, ptsDst.shape, MP.shape)

    # cv.imwrite("test_warp.png", perspect)
    # plt.imshow(cv.cvtColor(perspect, cv.COLOR_BGR2RGB))
    # plt.show()


# def read_and_process_camera_stream(camera_index):
#     cap = cv.VideoCapture(camera_index)
#
#     if not cap.isOpened():
#         raise ValueError("Cannot open camera")
#
#     while True:
#        frame = cap.read()
#         try:
#             # warped_frame = test_warp(frame)
#             cv.imshow('Warped Frame', frame)
#
#             # 按 'q' 键退出循环
#             if cv.waitKey(1) & 0xFF == ord('q'):
#                 break
#         except Exception as e:
#             print(f"Error processing frame: {e}")
#
#     cap.release()
#     cv.destroyAllWindows()
def run_opencv_camera():
    video_stream_path = 0  # local camera (e.g. the front camera of laptop)
    str_path: str = 'http://120.78.203.107:18080/ZTKJ/test.live.ts'
    cap = cv2.VideoCapture(str_path)

    while cap.isOpened():
        start = time.time()
        is_opened, frame = cap.read()
        frame_warp = test_warp(frame)
        # cv2.imshow('frame_warp', frame)
        cv2.imshow('frame_warp', frame_warp)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        end = time.time()
        print(f"代码运行时间：{(end - start) * 1000}ms")
    cap.release()


if __name__ == '__main__':
    # 读取网络视频流
    # str_path = 'http://120.78.203.107:18080/ZTKJ/test.live.ts'
    # read_and_process_camera_stream(str_path)

    run_opencv_camera()

    # img = cv2.imread("2.png")  # 读取彩色图像(BGR)
    # # test_warp(img)
    # cv2.imshow("1",img)
    # cv2.waitKey(0)