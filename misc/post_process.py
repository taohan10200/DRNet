import cv2 as cv
import numpy as np

class Processor():
    def __init__(self):
        self.kernel_2 = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

        self.kernel_3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        self.kernel_5 = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))

        self.min_area = 3
        self.tmp_result = {}
    def Dilation(self,binar_map):

        dilated = cv.dilate(binar_map,self.kernel_2)      #膨胀图像
        return  dilated

    def Erosion(self, binar_map):
        erosive = cv.erode(binar_map,self.kernel_3)        #腐蚀图像
        return erosive

    def morph_open(self, binar_map, iteration=1):

        # binar_map = cv.morphologyEx(binar_map, cv.MORPH_OPEN, self.kernel_5, iterations=iteration)
        for i in range(iteration):
            binar_map = cv.erode(binar_map, self.kernel_5)
            binar_map = cv.dilate(binar_map,  self.kernel_3)  # 膨胀图像


        return binar_map
    def connect_detection(self,area):
        cnt, labels, stats, centroids = cv.connectedComponentsWithStats(area, connectivity=4)  # centriod (w,h)
        if cnt == 1:
            self.tmp_result['num']=0
            return  True
        else:
            self.tmp_result['num']=cnt-1
            self.tmp_result['point']=centroids[1:,:]
            self.tmp_result['boxes']=stats[1:,:]
            return  True

    def Noise_box_detection(self,recs):
        maintain_list = []
        recs[:,2] = recs[:,0]+recs[:,2]
        recs[:, 3] = recs[:, 1] + recs[:,3]
        # print(recs)
        length = len(recs)

        for i in range(length):
            if i < length-1:   ##检测小框在大框内部
                j = i+1
                index = (recs[i][0]> recs[j:][:,0]) & (recs[i][1]> recs[j:][:,1])\
                        & (recs[i][2]<recs[j:][:,2]) & (recs[i][3]< recs[j:][:,3])
                index = np.where(index==True)[0]
                if index.size>0:
                    continue
                else:
                    maintain_list.append(i)
            else:
                maintain_list.append(i)
        return  maintain_list
    def get_boxInfo_from_Binar_map(self, Binar_numpy, pred_map, scale_map):
        Binar_numpy =Binar_numpy.squeeze().astype(np.uint8)

        pred_map = pred_map.squeeze()
        scale_map = scale_map.squeeze()

        print(Binar_numpy.max(),
              pred_map.max(),pred_map.sum())
        assert Binar_numpy.ndim == 2

        cnt, labels, stats, centroids = cv.connectedComponentsWithStats(Binar_numpy, connectivity=4)  # centriod (w,h)

        # print(cnt-1)
        boxes = stats[1:, :]
        points = centroids[1:, :]
        index = (boxes[:, 4] >= self.min_area)

        dict_1 = {}

        new_id = 0
        for id, bool in enumerate(index, 1):
            if bool:
                dict_1.update({new_id:id})
                new_id+=1

        boxes = boxes[index]
        points = points[index]


        # print(len(points))
        dist = np.sqrt(boxes[:,4])
        order = np.argsort(dist)

        dict_2 = {new_id: dict_1[old_id]  for (new_id, old_id) in enumerate(order, 0)}


        points = points[order]
        boxes = boxes[order]

        # print(boxes)
        assert  len(boxes)==len(points)
        # print('aaaa', len(boxes))
        maintain_list = self.Noise_box_detection(boxes.copy())

        boxes = boxes[maintain_list]
        points = points[maintain_list]

        dict_3 = {new_id: dict_2[old_id] for (new_id, old_id) in enumerate(maintain_list, 0)}

        assert  len(boxes) == len(points)

        # print('bbbb',len(boxes))
        iou_list, ratio_list = [], []
        boxes_app = []
        points_app = []
        for id in range(len(boxes)):

            if boxes[id, 4] > 10:

                w_s, h_s, w, h = boxes[id, 0], boxes[id, 1], boxes[id, 2], boxes[id, 3]
                iou = boxes[id, 4] / (w * h)
                ration = h / w
                if ration>2 or ration<0.5 or iou<0.75:
                    ratio_list.append(id)
                    sub_label = labels[h_s:h_s + h, w_s:w_s + w].copy()

                    mask = np.zeros_like(sub_label)
                    mask [sub_label == dict_3[id]] =1

                    sub_pre =  pred_map[h_s:h_s + h, w_s:w_s + w].copy()
                    sub_pre = (sub_pre * mask)
                    idx = (mask>0)
                    sub_pre = (sub_pre-sub_pre[idx].min())/(sub_pre[idx].max()-sub_pre[idx].min())
                    sub_binar = (sub_pre>0.5).astype(np.uint8)
                    self.connect_detection(sub_binar)

                    if self.tmp_result['num']>=1:
                        # print('pred_num', self.tmp_result['num'])
                        self.tmp_result['boxes'][:, 0] += w_s
                        self.tmp_result['boxes'][:, 1] += h_s
                        self.tmp_result['point'][:,0] += w_s
                        self.tmp_result['point'][:, 1] += h_s
                        boxes[id,:] = self.tmp_result['boxes'][0,:]
                        points[id,:] = self.tmp_result['point'][0,:]
                        for k in range(1,self.tmp_result['num']):
                            boxes_app.append(self.tmp_result['boxes'][k,:])
                            points_app.append(self.tmp_result['point'][k, :])

        print( len(iou_list), len(ratio_list))


        print('original:{}, add_boxes:{}, final_boxes:{}'.format(len(boxes), len(boxes_app), len(boxes)+len(boxes_app)))

        if len(boxes_app)>0:
            boxes = np.concatenate((boxes, np.array(boxes_app)))
            points = np.concatenate((points, np.array(points_app)))
            # print(boxes)

        if boxes.ndim ==1:
            boxes=boxes[np.newaxis,:]
        new_boxes = np.zeros((len(points), 5)).astype(np.float32)
        # print(boxes,points)
        for i in range(len(boxes)):
            x_s, y_s, w, h, area = boxes[i]
            pred = scale_map[y_s:y_s + h, x_s:x_s + w]
            mask = Binar_numpy[y_s:y_s + h, x_s:x_s + w]
            score = mask.sum() / (w * h)
            # scale = np.power(10, scale_map[int(points[i][1]), int(points[i][0])])
            # scale = np.power(10, pred[mask > 0].max())
            # scale = np.exp(min(np.log(8),  max(pred[mask>0].sum()/area - 1, 0)))
            scale = max(1, pred[mask > 0].max())
            # print(score, scale)
            add_w, add_h = w * scale - w, h * scale - h
            new_x_s, new_y_s = x_s - add_w // 2, y_s - add_h // 2
            new_x_e, new_y_e = x_s + (w + add_w // 2 + add_w % 2), y_s + (h + add_h // 2 + add_h % 2)
            new_boxes[i] = [new_x_s, new_y_s, new_x_e, new_y_e, score]

        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = boxes[:, :4]

        # print(boxes, new_boxes)
        # pdb.set_trace()
        pre_data = {'num': len(points), 'points': points, 'boxes': boxes, 'new_boxes': new_boxes}
        return pre_data

if __name__ == '__main__':
    from PIL import  Image
    import matplotlib.pyplot as plt
    import time
    s_t = time.time()
    img =  binar_map =  cv.imread('3114.jpg')
    binar_map =  cv.imread('3114_binar_map.jpg',cv.IMREAD_GRAYSCALE)
    pred_map = np.squeeze(np.load('3114_pred.npy', allow_pickle=True))

    # print(pred_map.shape)
    # print(binar_map.shape)
    # cv.imshow('aaa',pred_map)
    # cv.waitKey()
    ret, binar_map = cv.threshold(binar_map, 21,255, cv.THRESH_BINARY)
    binar_map = (pred_map>0.2).astype(np.uint8)

    processor= Processor()

    pre_data = processor.get_boxInfo_from_Binar_map(binar_map,pred_map,pred_map)
    print(pre_data['num'])

    pred_map = (pred_map*255).astype(np.uint8)
    binar_map = (binar_map * 255).astype(np.uint8)
    binar_map=cv.applyColorMap(binar_map, cv.COLORMAP_JET)
    pred_map = cv.applyColorMap(pred_map, cv.COLORMAP_JET)


    point_color = (0, 255, 0)  # BGR
    thickness = 1
    lineType = 4
    for i, box in enumerate(pre_data['boxes'], 0):
        wh_LeftTop = (box[0], box[1])
        wh_RightBottom = ( box[2],box[3])
        cv.rectangle(binar_map, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)
        cv.rectangle(img, wh_LeftTop, wh_RightBottom, point_color, thickness, lineType)


    binar_show=Image.fromarray(cv.cvtColor(binar_map, cv.COLOR_BGR2RGB))
    img_show = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    pred_show = Image.fromarray(cv.cvtColor(pred_map, cv.COLOR_BGR2RGB))
    # ero_binar_show = Image.fromarray(cv.cvtColor(ero_binar_map, cv.COLOR_BGR2RGB))

    #
    binar_show.show(title='ori_binar_map')
    img_show.show(title='img')
    pred_show.show(title='pred')
    # ero_binar_show.show(title='ero_binar_map')


    print("time:{}".format(time.time()-s_t))