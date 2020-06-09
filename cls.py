# -*- coding:utf-8 -*-

import fasttext

qa_cls_train_data_path = 'data/qa_cls_train_data.txt'
qa_cls_test_data_path = 'data/qa_cls_test_data.txt'
qa_cls_model_path = 'model/qa_cls_model.bin'
stop_words_file_path = 'data/stopwords.txt'
stop_words = open(stop_words_file_path, 'r').readlines()
stop_words = [word.strip() for word in stop_words]

def train_model():
    model = fasttext.train_supervised(
        qa_cls_train_data_path,
        epoch=25,
        lr=1,
        wordNgrams=3,
        dim=100,
        loss='hs'
    )
    model.save_model(qa_cls_model_path)
    print(model.test(qa_cls_test_data_path))


def cls_predict():
    model = fasttext.load_model(qa_cls_model_path)
    text = '刘翔出生地在哪儿'
    s = list(text)
    s = [ts for ts in s if ts not in stop_words]
    text = ' '.join(s)
    print(model.predict(text))
    # for line in open(qa_cls_test_data_path).readlines():
    #     line = line.strip('\n').split('\t')[1]
    #     s = list(line)
    #     s = [ts for ts in s if ts not in stop_words]
    #     text = ' '.join(s)
    #     print(line, model.predict(text))


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
