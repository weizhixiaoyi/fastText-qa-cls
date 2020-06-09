# -*- coding:utf-8 -*-

import fasttext

qa_cls_train_data_path = 'data/qa_cls_train_data.txt'
qa_cls_test_data_path = 'data/qa_cls_test_data.txt'
qa_cls_model_path = 'model/qa_cls_model.bin'


# qa_cls_train_data_path = 'data/template/cooking.train'
# qa_cls_test_data_path = 'data/template/cooking.valid'
# qa_cls_model_path = 'model/cooking_model.bin'


def train_model():
    model = fasttext.train_supervised(qa_cls_train_data_path, epoch=25, lr=1, wordNgrams=3, dim=100, loss='hs')
    model.save_model(qa_cls_model_path)
    print(model.test(qa_cls_test_data_path))


def cls_predict():
    model = fasttext.load_model(qa_cls_model_path)
    text = '刘翔出生地在哪儿'
    text = ' '.join(list(text))
    print(model.predict(text))
    # for line in open(qa_cls_test_data_path).readlines():
    #     line = line.strip('\n').split('\t')[1]
    #     print(line, model.predict(line))


if __name__ == '__main__':
    mode = 'train'
    # mode = 'predict'
    # mode = 'train and predict'
    if mode == 'train':
        train_model()
    elif mode == 'predict':
        cls_predict()
    else:
        train_model()
        cls_predict()
