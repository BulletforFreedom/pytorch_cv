import numpy as np


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, x2, y2]
    boxes: [boxes_count, (x1, y1, x2, y2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    Attention: input must be a np_array
    """
    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area - intersection
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, x2, y2)].

    For better performance, pass the smaller set first and the larger second.
    """
    # Areas of anchors and GT boxes
    area_b1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_b2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[0]):
        overlaps[i, :] = compute_iou(boxes1[i], boxes2, area_b1[i], area_b2)
    return overlaps


def compute_ap_cat(gt_boxes, gt_class_ids,
                   pred_boxes, pred_class_ids, pred_scores,
                   iou_threshold=0.5):
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    index = np.argsort(pred_scores)[::-1]
    pred_class_ids = pred_class_ids[index]
    pred_boxes = pred_boxes[index]
    pred_scores = pred_scores[index]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    iou = compute_overlaps(gt_boxes, pred_boxes)
    # Loop through ground truth boxes and find matching predictions
    pred_match = np.zeros((iou.shape[1]))
    match_count = 0
    gt_match = np.zeros((iou.shape[0]))
    for i in range(iou.shape[1]):
        # Find best matching ground truth box
        sorted_index = np.argsort(iou[:, i])[::-1]
        for j in sorted_index:
            if gt_match[j] == 1:
                continue
            if iou[j, i] < iou_threshold:
                break
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break
    # Compute precision and recall at each prediction box step
    recalls = np.cumsum(pred_match).astype(np.float32) / gt_boxes.shape[0]
    precisions = np.cumsum(pred_match).astype(np.float32) / \
        (np.arange(len(pred_match)) + 1)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1])
                 * precisions[indices])

    return mAP, precisions, recalls, iou


def compute_map(gt_boxes, gt_class_ids,
                pred_boxes, pred_class_ids, pred_scores,
                iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    FP_cls = 0
    FP_bg = 0
    FN = 0
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    index = np.argsort(pred_scores)[::-1]
    pred_class_ids = pred_class_ids[index]
    pred_boxes = pred_boxes[index]
    pred_scores = pred_scores[index]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    iou = compute_overlaps(gt_boxes, pred_boxes)
    # Loop through ground truth boxes and find matching predictions
    pred_match = np.zeros((iou.shape[1]))
    match_count = 0
    gt_match = np.zeros((iou.shape[0]))
    for i in range(iou.shape[1]):
        # Find best matching ground truth box
        sorted_index = np.argsort(iou[:, i])[::-1]
        for j in sorted_index:
            if iou[j, i] < iou_threshold:
                FP_bg += 1
                break
            if pred_class_ids[i] != gt_class_ids[j]:
                FP_cls += 1
            if gt_match[j] == 1:
                continue
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = 1
                pred_match[i] = 1
                break
    FN = int(len(gt_match) - np.sum(gt_match))
    F = {'FP_bg': FP_bg, 'FP_cls': FP_cls, 'FN': FN}
    # Compute precision and recall at each prediction box step
    recalls = np.cumsum(pred_match).astype(np.float32) / gt_boxes.shape[0]
    precisions = np.cumsum(pred_match).astype(np.float32) / \
        (np.arange(len(pred_match)) + 1)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[1], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1])
                 * precisions[indices])

    return mAP, F, precisions, recalls, iou


def match_predict(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    pred_match: List of flag with matched_instances set one
    pred_score: List of score with matched_instance
    """
    # Trim zero padding and sort predictions by score from high to low
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]
    index = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[index]
    pred_scores = pred_scores[index]

    # Compute IoU overlaps [pred_boxes, gt_boxes]
    iou = compute_overlaps(gt_boxes, pred_boxes)
    # Loop through ground truth boxes and find matching predictions
    pred_match = np.zeros((iou.shape[1]))
    match_count = 0
    gt_match = np.zeros((iou.shape[0]))
    for i in range(iou.shape[1]):
        # Find best matching ground truth box
        sorted_index = np.argsort(iou[:, i])[::-1]
        for j in sorted_index:
            if gt_match[j] == 1:
                continue
            if iou[j, i] < iou_threshold:
                break
            gt_match[j] = 1
            pred_match[i] = 1
            break

    return [pred_match, pred_scores]


def compute_map_cat(gt_boxes_cat, pred_boxes_cat, pred_scores_cat, cat_num, iou_threshold):
    result_match_cat = {i: [] for i in range(cat_num + 1)}
    result_scores_cat = {i: [] for i in range(cat_num + 1)}
    gt_count = {i: 0 for i in range(cat_num + 1)}
    FP_cls = np.zeros([cat_num + 1])
    FP_bg = np.zeros([cat_num + 1])
    FN = np.zeros([cat_num + 1])

    for i in range(len(pred_scores_cat)):
        for j in range(1, cat_num + 1, 1):
            if ((j in gt_boxes_cat[i + 1]) and (j in pred_boxes_cat[i + 1])):
                gt_count[j] += len(gt_boxes_cat[i + 1][j])
                tmp = match_predict(gt_boxes_cat[i + 1][j],
                                    pred_boxes_cat[i +
                                                   1][j], pred_scores_cat[i + 1][j],
                                    iou_threshold=iou_threshold)
                result_match_cat[j] += list(tmp[0])
                result_scores_cat[j] += list(tmp[1])

    recalls = {i + 1: [] for i in range(cat_num)}
    precisions = {i + 1: [] for i in range(cat_num)}
    AP = {i + 1: 0 for i in range(cat_num)}
    # print(gt_count)
    for i in range(cat_num):
        result_match_cat[i + 1] = np.array(result_match_cat[i + 1])
        result_scores_cat[i + 1] = np.array(result_scores_cat[i + 1])
        index = np.argsort(result_scores_cat[i + 1])[::-1]

        result_match_cat[i + 1] = result_match_cat[i + 1][index]
        result_scores_cat[i + 1] = result_scores_cat[i + 1][index]

        # Compute precision and recall at each prediction box step

        recalls[i +
                1] = np.cumsum(result_match_cat[i + 1]).astype(np.float32) / gt_count[i + 1]
        precisions[i + 1] = np.cumsum(result_match_cat[i + 1]).astype(np.float32) / \
            (np.arange(len(result_match_cat[i + 1])) + 1)

        # Pad with start and end values to simplify the math
        precisions[i + 1] = np.concatenate([[1], precisions[i + 1], [0]])
        recalls[i + 1] = np.concatenate([[0], recalls[i + 1], [1]])

        # Ensure precision values decrease but don't increase. This way, the
        # precision value at each recall threshold is the maximum it can be
        # for all following recall thresholds, as specified by the VOC paper.
    # print(precisions[1])
    # print('--------------------')
    # print(recalls[1])
        for j in range(len(precisions[i + 1]) - 2, -1, -1):
            precisions[i + 1][j] = np.maximum(precisions[i + 1]
                                              [j], precisions[i + 1][j + 1])

        # Compute mean AP over recall range
        indices = np.where(recalls[i + 1][:-1] != recalls[i + 1][1:])[0] + 1
        AP[i + 1] = np.sum((recalls[i + 1][indices] - recalls[i + 1][indices - 1])
                           * precisions[i + 1][indices])
    mAP = 0
    cot = 0
    for i in range(cat_num):
        mAP += AP[i + 1]
        if AP[i + 1] != 0:
            cot += 1

    return mAP / cot


def trim_zeros(x):
    assert len(x.shape) == 2
    x = x[~np.all(x == 0, axis=1)]
    return x


def fuckyou_trans(i):
    if i == 1:
        return 4
    elif i == 2:
        return 2
    elif i == 3:
        return 1
    elif i == 4:
        return 6
    elif i == 5:
        return 3
    elif i == 6:
        return 7
    elif i == 7:
        return 5

# a = np.array([1, 3, 4, 5, 6])
# index = np.argsort(a)[::-1]
# print(a[index])
