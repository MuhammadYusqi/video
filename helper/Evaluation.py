import numpy as np

def NMS_One(boxes, probs=None, overlapThresh=0.3):
    
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")

def NMS(boxes, prob, th=0.3):
    result = []
    for b,p in zip(boxes,prob):
        b = np.array(b)
        nms = NMS_One(b, p, th)
        result.append(nms)
    result = np.array(result)

    return result

def IoU(predict,target):
    Xmin1, Ymin1, Xmax1, Ymax1 = predict
    Xmin2, Ymin2, Xmax2, Ymax2 = target  

    Xmin_i = max(Xmin1, Xmin2)
    Ymin_i = max(Ymin1, Ymin2)
    Xmax_i = min(Xmax1, Xmax2)
    Ymax_i = min(Ymax1, Ymax2)

    w_i = Xmax_i - Xmin_i
    h_i = Ymax_i - Ymin_i
    if w_i <= 0 or h_i <= 0 :
        return 0
    Intersection = w_i * h_i

    w1 = Xmax1 - Xmin1
    h1 = Ymax1 - Ymin1
    w2 = Xmax2 - Xmin2
    h2 = Ymax2 - Ymin2

    Union = (w1 * h1) + (w2 * h2) - Intersection
    IoU = Intersection / Union

    return IoU

def Accuracy(annotations, boxes):
    conf_th = 0.3
    sum_tp, sum_fp, sum_fn = 0, 0, 0
    for anns, boxs in zip(annotations, boxes):
        tp = np.zeros((len(anns)))
        fp = 0

        for i, box in enumerate(boxs):
            ious = []
            
            for ann in anns:
                iou = IoU(box, ann)
                ious.append(iou)
            ious = np.array(ious)
            
            mask = ious >= conf_th
            tp[mask] += 1
            if not mask.any():
                fp += 1
        fn = len(tp[tp == 0])
        tp = sum(tp)
        
        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    recall = sum_tp / (sum_tp + sum_fn) 
    precision = sum_tp / (sum_tp + sum_fp) 
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1
