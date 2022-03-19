def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text = [], restore_transform=None,
                            id0=None,id1=None, scores=None
                            ):
    image0 = np.array(restore_transform(image0))
    image1 = np.array(restore_transform(image1))
    image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
    H0, W0, C = image0.shape
    H1, W1, C = image1.shape
    pre_inflow = np.zeros((H0, W0, 3)).astype(np.uint8)
    pre_outflow = np.zeros((H1, W1, 3)).astype(np.uint8)

    H, W = 2*max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W, C), np.uint8)
    out[:H0, :W0,:] = image0
    out[:H1, W0+margin:,:] = image1
    # out = np.stack([out]*3, -1)
    # import pdb
    # pdb.set_trace()
    out_by_point = out.copy()
    point_r_value = 10
    thickness = 3
    white = (255, 255, 255)
    RoyalBlue1 = np.array([255, 118, 72])  # np.array([205,82,180])
    red = [0, 0, 255]
    green = [0, 255, 0]
    blue = [255, 0, 0]
    pre_inflow[:, :, 0:3] = RoyalBlue1
    pre_outflow[:, :, 0:3] = RoyalBlue1

    kernel = 8
    wide = 2 * kernel + 1

    # ===================begin: inflow outflow map ================
    pre_outflow_p = kpts0[scores[:-1, -1] > 0.2]
    scores_= scores[:-1, -1][scores[:-1, -1] > 0.2]
    for row_id, (pos,s) in enumerate(zip(pre_outflow_p, scores_), 0):
        w, h = pos.astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(H0, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(W0, w + kernel + 1)
        red_ = [0,0,int(255*s)]
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red_)

        pre_outflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - H0),
                                                max(kernel - w, 0):wide - max(0, kernel + 1 + w - W0)]
    # ================================pre_inflow========================
    pre_inflow_p = kpts1[scores[-1, :-1] > 0.2]
    scores_ = scores[-1, :-1][scores[-1, :-1] > 0.2]
    for column_id, (pos,s) in enumerate(zip(pre_inflow_p,scores_), 0):
        w, h = pos.astype(np.int64)
        h_min, h_max = max(0, h - kernel), min(H0, h + kernel + 1)
        w_min, w_max = max(0, w - kernel), min(W0, w + kernel + 1)
        red_ = [0, 0, int(255 * s)]
        mask = generate_cycle_mask(kernel, kernel, RoyalBlue1, red_)
        pre_inflow[h_min:h_max, w_min:w_max] = mask[max(kernel - h, 0):wide - max(0, kernel + 1 + h - H0),
                                               max(kernel - w, 0):wide - max(0, kernel + 1 + w - W0)]
    out[H0:, :W0, :] = pre_outflow
    out[H1:, W0 + margin:, :] = pre_inflow
    #===================end: inflow outflow map ================

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        for x, y in kpts0:
            cv2.circle(out, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x, y), point_r_value, red, thickness, lineType=cv2.LINE_AA)

        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), point_r_value, red, thickness,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 3, white, -1, lineType=cv2.LINE_AA)

            cv2.circle(out_by_point, (x + margin + W0, y), point_r_value, blue, thickness,
                       lineType=cv2.LINE_AA)

        if id0 is not  None:
            for i, (id, centroid) in enumerate(zip(id0, kpts0)):
                cv2.putText(out, str(id), (centroid[0],centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if id1 is not None:
            for i, (id, centroid) in enumerate(zip(id1, kpts1)):
                cv2.putText(out, str(id), (centroid[0]+margin+W0, centroid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

        cv2.circle(out_by_point, (x0, y0), point_r_value, green, thickness, lineType=cv2.LINE_AA)
        cv2.circle(out_by_point, (x1 + margin + W0, y1), point_r_value, green, thickness,
                   lineType=cv2.LINE_AA)

    Ht = int(H*10 / 480)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    text_S_H = 65
    cv2.putText(out, 'previous frame()', (30, 30 ), cv2.FONT_HERSHEY_DUPLEX,
                H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
    for i, t in enumerate(text):
        if i == 0:
            cv2.putText(out, t, (W0-600, H0+text_S_H), cv2.FONT_HERSHEY_DUPLEX,
                        H*1.2/1080, txt_color_fg, 2, cv2.LINE_AA)
        if i == 1:
            cv2.putText(out, t, (30, H0 + text_S_H), cv2.FONT_HERSHEY_DUPLEX,
                        H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
        if i == 2:
            cv2.putText(out, t, (2*W0-800, H0 + text_S_H), cv2.FONT_HERSHEY_DUPLEX,
                        H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
        if i == 3:
            cv2.putText(out, t, (30, H0 + int(text_S_H*2.5)), cv2.FONT_HERSHEY_DUPLEX,
                        H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)

        if i == 4:
            cv2.putText(out, t, (2*W0-800, H0 + int(text_S_H*2.5)), cv2.FONT_HERSHEY_DUPLEX,
                        H * 1.2 / 1080, txt_color_fg, 2, cv2.LINE_AA)
        # cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
        #             H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)
        # cv2.putText(out_by_point, t, (10, Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
        #         H * 1.0 / 480, txt_color_fg, 1, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), out)
        cv2.imwrite(str('point_'+path), out_by_point)
    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out,out_by_point