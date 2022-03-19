import cv2
def getFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # cv2.imshow('video', frame)
                numFrame += 1
                print('frame:', numFrame)
                newPath = svPath + str(numFrame) + ".jpg"
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        # if cv2.waitKey(10) == 27:
        #     break

if __name__ == '__main__':

    import os
    import os.path as osp
    root = '/data/GJY/ht/CVPR2022/dataset/demo_video'
    videoPath = osp.join(root,'VID_20210423_145537.mp4')
    savePicturePath = osp.join(root, 'beijingxi/')
    os.makedirs(savePicturePath,mode=0o777, exist_ok=True)
    getFrame(videoPath, savePicturePath)