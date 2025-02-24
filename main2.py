import cv2
import numpy as np
import matplotlib.pyplot as plt
import time






# 定义原图中的四个点（根据图像的倾斜选择）
# 这些点应按顺时针顺序给出：左上角，右上角，右下角，左下角
src_points = np.float32([
    [412, 203],  # 左上
    [569, 202],  # 右上
    [612, 465],  # 右下
    [166, 463]  # 左下
])
# [387, 151],  # 左上
# [591, 142],  # 右上
# [506, 562],  # 右下
# [30, 563]  # 左下
# 定义目标图像中的四个点（正射图像的四个角）
# 这里我们假设目标图像是一个矩形
dst_points = np.float32([
    [0, 0],  # 左上
    [400, 0],  # 右上
    [400, 850],  # 右下
    [0, 850]  # 左下
])


def show_point_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)


def input(url: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(url)

#优化方向
# 1:透视变换只进行一次
# 2:在循环中同时展示原图像和变换图像

def correct_perspective(img, src_points, dst_points, end_point):
    # 读取图像

    # 获取透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 进行透视变换，生成正射图像

    result = cv2.warpPerspective(img, matrix, end_point.astype(np.int32))

    return result


def show_image(image, title='Image'):
    # 显示图像
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def image_warp():
    # 输入图像路径
    image_path = '5.png'  # 修改为你的图片路径

    # 进行透视变换
    result_image = correct_perspective(
        cv2.imread(image_path),
        src_points,
        dst_points)

    # 显示原图和变换后的图像
    original_image = cv2.imread(image_path)
    show_image(original_image, 'Original Image')
    show_image(result_image, 'Corrected Perspective')

    # 保存结果
    cv2.imwrite('corrected_perspective.jpg', result_image)


if __name__ == '__main__':
    url_online = 'http://120.78.203.107:18080/ZTKJ/test.live.ts'
    url_offline = r'E:\VideoTwin\Code\pythonProject\test.live.ts'

    video = input(url_online)
    # video = input(url_offline)
    if not video.isOpened(): raise "video cannot open"
    #
    # cv2.imshow('Frame', np.zeros((480, 640, 3)))
    # cv2.setMouseCallback('Frame', show_point_event)

    while video.isOpened():
        start = time.time()
        ret, frame = video.read()
        if not ret:
            raise ('video closed')
        cv2.imshow('WarpedFrame', correct_perspective(frame, src_points, dst_points, dst_points[2, 0:2]))
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        end = time.time()
        print(f"代码运行时间：{(end - start) * 1000}ms")
    video.release()
    cv2.destroyAllWindows()
