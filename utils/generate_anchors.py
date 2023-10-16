# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import pdb

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])


# 给定anchor信息，获得不同比例anchor信息
def _scale_enum(anchor, scales):
    """
    为每个标度列出一组锚点
    """
    # 获得宽、高、中心点坐标
    w, h, x_ctr, y_ctr = _whctrs(anchor)

    # 将宽高按不同比例放缩
    ws = w * scales
    hs = h * scales
    # print('w * scales = ws:  {} * {} = {}'.format(w, scales, ws))
    # print('h * scales = hs:  {} * {} = {}'.format(h, scales, hs))

    # 计算放缩后的框，对应左上角、右下角坐标
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# 将(x1、y1、x2、y2) -> (w、h、mid_x、mid_y)
def _whctrs(anchor):
    """
    返回锚点(窗口)的宽度、高度、x中心和y中心
    anchor :[0 0 15 15]
    """
    # （x2 - x1）宽：15 + 1 = 16
    w = anchor[2] - anchor[0] + 1
    # （y2 - y1）长：15 + 1 = 16
    h = anchor[3] - anchor[1] + 1
    # （x1 + w//2）中心x坐标
    x_ctr = anchor[0] + 0.5 * (w - 1)
    # （y1 + h//2）中心y坐标
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

# 给定长、宽和中心点坐标，获得anchor左上角和右下角坐标
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    给定围绕中心(x_ctr, y_ctr)的宽度(ws)和高度(hs)向量，输出一组锚点(窗口)
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    #消除负数坐标
    for anchor in anchors:

        x1=anchor[0]
        y1=anchor[1]
        if x1 <0:
            anchor[0]=0
            anchor[2]=anchor[2]+np.abs(x1)

        if y1<0:
            anchor[1]=0
            anchor[3]=anchor[3]+np.abs(y1)

    return anchors

# 获得anchor
def _ratio_enum(anchor, ratios):
    """
    为每个长宽比列出一组锚点
    anchor :[0  0 15 15]
    ratios : [0.5, 1, 2]
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    # anchor面积 16x16=256
    size = w * h
    # anchor不同比例面积 [512. 256. 128.]
    size_ratios = size / ratios
    # 宽 [23. 16. 11.]
    ws = np.round(np.sqrt(size_ratios))
    # 长 [12. 16. 22.]
    hs = np.round(ws * ratios)
    # 有了宽、高、中心点，就可以获得锚框
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(cordi=[0,0],base_size=[31,31], ratios=[0.5, 1, 2],
                     scales=2**np.arange(1,2)):
    """
    anchor的基本大小设定为16x16
    scales = [ 8 16 32 ]
    """
    # [ 0  0 15 15 ]
    # base_anchor = np.array([1, 1, base_size, base_size]) - 1
    x1=cordi[0]
    y1=cordi[1]
    base_anchor = np.array([x1, y1, x1+base_size[0], y1+base_size[1]])
    
    # [[-3.5  2.  18.5 13. ]
    #  [ 0.   0.  15.  15. ]
    #  [ 2.5 -3.  12.5 18. ]]
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])

    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(a)
