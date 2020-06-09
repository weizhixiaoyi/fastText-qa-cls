# -*- coding:utf-8 -*-
import jieba
from sklearn.model_selection import train_test_split

file_path_1 = 'nlu1.txt'
file_path_2 = 'nlu2.txt'
stop_words_file_path = 'stopwords.txt'
stop_words = open(stop_words_file_path, 'r').readlines()
stop_words = [word.strip() for word in stop_words]


def process(data):
    new_data = []
    label = ''
    for line in data:
        line = line.strip('\n')
        if 'intent' in line:
            label = line.split('intent')[-1].strip(':').lower()
            # print(label)
            continue
        if line is None:
            continue

        line = line[2:]
        new_data.append('__label__' + label + '\t' + line)
    return new_data


file1_data = open(file_path_1, "r").readlines()
file2_data = open(file_path_2, "r").readlines()

file1_data_process = process(file1_data)
file2_data_process = process(file2_data)
file_data = list(set(file1_data_process + file2_data_process))
train_file_data, test_file_data = train_test_split(file_data, test_size=0.2)
print(len(train_file_data))
print(len(test_file_data))


def save_data(file_data, file_path):
    with open(file_path, 'w') as f:
        temp_dict = {}
        for line in file_data:
            l, s = line.split('\t')[0], line.split('\t')[1]
            if l in temp_dict.keys():
                temp_dict[l].append(s)
            else:
                temp_dict[l] = [s]

        for k, v in temp_dict.items():
            for vv in v:
                # word
                s = list(vv)
                ss = ' '.join([ts for ts in s if ts not in stop_words])

                # words
                # t = jieba.lcut(vv)
                # tt = ' '.join(t)
                f.write(k + '\t' + ss + '\n')


save_data(train_file_data, 'qa_cls_train_data.txt')
save_data(test_file_data, 'qa_cls_test_data.txt')
