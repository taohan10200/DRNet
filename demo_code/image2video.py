import os
import cv2
from PIL import Image

def unlock_movie(path):
    """ 将视频转换成图片
    path: 视频路径 """
    cap = cv2.VideoCapture(path)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    while suc:
      frame_count += 1
      suc, frame = cap.read()
      params = []
      params.append(2)  # params.append(1)
      cv2.imwrite('frames\\%d.jpg' % frame_count, frame, params)

    cap.release()
    print('unlock movie: ', frame_count)


def  jpg_to_video(img_path,video_path, fps,size):
    """ 将图片合成视频. path: 视频路径，fps: 帧率 """

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    images = os.listdir(img_path)#os.listdir()方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
    images.sort()
    images.sort(key=lambda x: int(x.split('_')[0]))
    print(images)
    image = Image.open(os.path.join(img_path,images[0]))

    vw = cv2.VideoWriter(video_path, fourcc, fps,size)

# os.chdir('frames')
    for file in images:
        imagefile = os.path.join(img_path, file)
        try:
            new_frame = cv2.imread(imagefile)
            new_frame=cv2.resize(new_frame, size, interpolation=cv2.INTER_AREA)
            # img_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
            # cv2.putText(new_frame, text, (int(10), int(450 + 15)), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255),2)
            vw.write(new_frame)

            print(imagefile)
        except Exception as exc:
            print(imagefile, exc)
    vw.release()
    print(video_path, 'Synthetic success!')



if __name__ == '__main__':

    root_path =  '/data/GJY/ht/CVPR2022/dataset/demo_video_HT'
    img_path = os.path.join(root_path, 'pred')


    size=(1920,1080)
    size = (int((1920+1600)/3),int((1130)/3))
    video_path = os.path.join(root_path,'beijing_pred_'+str(size[0])+ '_'+str(size[1])+'.avi')
    jpg_to_video(img_path,video_path, 25,size)  # 图片转视频