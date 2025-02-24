import cv2
import time


def input_video(url: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(url)


if __name__ == '__main__':
    # url_online = 'http://120.78.203.107:18080/ZTKJ/test.live.ts'
    # url_offline = r'E:\VideoTwin\Code\pythonProject\test.live.ts'
    str_path1: str = 'http://120.78.203.107:18080/ZTKJ/test.live.ts'  # 720*1280
    str_path2: str = 'http://120.78.203.107:18080/ZTKJ/qiangji_test2.live.ts'  # 1080*1920
    str_path3: str = 'http://120.78.203.107:18080/ZTKJ/qouji_test.live.ts'  # 1080*1920
    video1 = input_video(str_path1)
    video2 = input_video(str_path2)
    video3 = input_video(str_path3)

    # video = input(url_offline)
    if not video1.isOpened() or not video2.isOpened() or not video3.isOpened(): raise "video cannot open"
    #
    # cv2.imshow('Frame', np.zeros((480, 640, 3)))
    # cv2.setMouseCallback('Frame', show_point_event)

    while video1.isOpened():
        start = time.time()
        ret1, frame1 = video1.read()
        frame1 = cv2.resize(frame1, (1920, 1080))
        if not ret1:
            raise ('video1 closed')

        ret2, frame2 = video2.read()
        if not ret2:
            raise ('video2 closed')

        ret3, frame3 = video3.read()
        if not ret3:
            raise ('video3 closed')

        frame = cv2.hconcat([frame2, frame1, frame3])
        cv2.imshow('Frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break
        end = time.time()
        print(f"代码运行时间：{(end - start) * 1000}ms")

    cv2.destroyAllWindows()
