"""
@File : 视频裁减.py

@Author: sivan Wu

@Date : 2020/1/21 11:24 上午

@Desc : 视频裁减成一张张图

"""
import os
import cv2
import face_recognition
# VIDEO_PATH = "/Users/yirufeng/Downloads/YourName.mkv"
# VIDEO_PATH2 = "../data/其他/真人视频.mp4"
VIDEO_PATH = "./001.mp4"  ## 4分32秒


output_path = "./imgs"





# 如果路径不存在则创建路径car
if not os.path.exists(output_path):
    os.mkdir(output_path)

cap = cv2.VideoCapture(VIDEO_PATH)
count = 0
# 检查是否成功初始化摄像机
while cap.isOpened():
    ret, frame = cap.read()
    # 说明正常读取还没有结束
    if ret == True:
        # cv2.imshow("frame", frame)
        # 检测是否有人脸
        if len(face_recognition.face_locations(frame)):
            cv2.imwrite(output_path + "/" + str(count) + ".png", frame)
            count += 1
            print("保存第", count, "张图")
        if cv2.waitKey(280) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
