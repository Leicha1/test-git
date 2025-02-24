import cv2
import time
import numpy as np


# src_points = np.float32([
src_points = np.int32([
    [412, 203],  # 左上
    [569, 202],  # 右上
    [612, 465],  # 右下
    [166, 463]  # 左下
])
def run_opencv_camera():
    video_stream_path = 0  # local camera (e.g. the front camera of laptop)
    str_path1: str = 'http://120.78.203.107:18080/ZTKJ/test.live.ts'           # 720*1280
    str_path2: str = 'http://120.78.203.107:18080/ZTKJ/qiangji_test2.live.ts'  # 1080*1920
    str_path3: str = 'http://120.78.203.107:18080/ZTKJ/qouji_test.live.ts'     # 1080*1920
    cap = cv2.VideoCapture(str_path1)
    # cap = cv2.VideoCapture(str_path2)
    # cap = cv2.VideoCapture(str_path3)
    # cap = cv2.VideoCapture(0)

    while cap.isOpened():
        start = time.time()
        is_opened, frame = cap.read()

        # cv2.imwrite('1_show.png', frame)
        print(frame.shape)
        # 显示图像
        cv2.imshow('frame_warp', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        end = time.time()
        print(f"代码运行时间：{(end - start) * 1000}ms")
    cap.release()


if __name__ == '__main__':
    run_opencv_camera()
    # 绘制多边形（四边形）
    # image = cv2.imread('1_show.png')
    # cv2.polylines(image, [src_points], isClosed=True, color=(0, 255, 0), thickness=2)
    # # cv2.imshow('image', image)
    # cv2.imwrite('1_show_target.png', image)
    # cv2.waitKey(0)