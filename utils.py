# *_*coding:utf-8 *_*

def f1_score(true_path, predict_path, lengths):
    batch_TP_FP = 0
    batch_TP_FN = 0  # not division by zero
    batch_TP = 0
    for true, predict, len in zip(true_path, predict_path, lengths):
        true = true[:len]  # remove the 0 padding

        TP_and_FP = 0
        TP_FN = 0
        TP = 0
        for i in predict:
            if i >= 3 and i % 2 == 1:
                TP_and_FP += 1

        for i in true:
            if i >= 3 and i % 2 == 1:
                TP_FN += 1

        for i, index in enumerate(true):
            if predict[i] == index and index != 2:
                if index >= 3 and index % 2 == 1:
                    TP += 1
        # print('------')
        # print(TP)
        # print(TP_FP)
        # print(TP_FN)
        # print(predict)
        # print(true)

        batch_TP_FP += TP_and_FP
        batch_TP_FN += TP_FN
        batch_TP += TP

    if batch_TP_FP == 0:
        precision = 0
    else:
        precision = batch_TP / batch_TP_FP
    if batch_TP_FN == 0:
        recall = 0
    else:
        recall = batch_TP / batch_TP_FN
    if batch_TP == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    print(f'precision: {precision:.2f}, '
          f'recall: {recall:.2f}, f1:{f1:.2f}')
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write(f'precision: {precision:.2f}, '
          f'recall: {recall:.2f}, f1:{f1:.2f}')
