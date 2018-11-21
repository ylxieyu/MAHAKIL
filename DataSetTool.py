import numpy as np
import os
import Tools.FilesTool as FilesTool
from MAHAKIL.mahakil import MAHAKIL


class DataSetTool:
    # 08版的度量补偿
    # Mij in Target = (Mij in Target * Mean(Mj in Source)) / Mean(Mj) in  Target
    @staticmethod
    def metric_compensation(source, target):
        # 遍历每一个度量属性
        for j in range(target.shape[1]):
            # 计算每个度量属性的均值
            metric_mean_source = np.mean(source[:, j])
            metric_mean_target = np.mean(target[:, j])
            # 遍历每一个样例
            for i in range(target.shape[0]):
                target[i, j] = (target[i, j] * metric_mean_source) / metric_mean_target
        return target

    # 17版进行调整的度量补偿
    # Mij in Source = (Mij in Source * Mean(Mj in Target)) / Mean(Mj) in Source
    @staticmethod
    def metric_compensation_adopt(source, target):
        # 遍历每一个度量属性
        for j in range(source.shape[1]):
            # 计算每个度量属性的均值
            metric_mean_source = np.mean(source[:, j])
            metric_mean_target = np.mean(target[:, j])
            # 遍历每一个样例
            for i in range(source.shape[0]):
                source[i, j] = (source[i, j] * metric_mean_target) / metric_mean_source
        return source

    # 读取文件夹下的所有文件，并返回处理好的数据集
    # metrics_num 度量数目（原始数据中除开标签列的列数）txt文件读取时需要
    # is_sample 是否重采样
    # is_normalized 是否数据归一化
    @staticmethod
    def init_data(folder_path, metrics_num=20, is_sample=True, is_normalized=True):
        # 获取目录下所有原始文件
        files = os.listdir(folder_path)
        data_list, label_list = [], []
        for file in files:
            # 每一个子文件的真实路径
            file_path = folder_path+file
            # txt文件
            if 'txt' == FilesTool.file_type(file) or 'TXT' == FilesTool.file_type(file):
                # 直接读取文件
                data_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=range(0, metrics_num+1))
                label_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=metrics_num+1)
                if is_normalized:
                    # 数据归一化
                    data_file -= data_file.min()
                    data_file /= data_file.max()
                    label_file -= label_file.min()
                    label_file /= label_file.max()
                # 加入列表
                data_list.append(data_file)
                label_list.append(label_file)
            # arff文件
            if 'arff' == FilesTool.file_type(file) or 'ARFF' == FilesTool.file_type(file):
                relation, attribute, data = [], [], []  # 保存arff 中的信息
                class_identify, t_class_identify, f_class_identify = [], '', ''
                is_first = True
                with open(file_path, 'r') as arff_file:
                    lines = arff_file.readlines()
                    for line in lines:
                        if '@relation' in line:
                            r = line.split(' ')
                            relation.append(tuple([r[0].replace('@', '').strip(), r[1].strip()]))
                            continue
                        if '@attribute' in line:
                            attribute.append(line.replace('@attribute', '').strip())
                            continue
                        if '\n' == line or '@data' in line:
                            continue
                        else:
                            if is_first:
                                # 如果第一次进来，通过判断已经读取的属性，将标签转化问1、0
                                class_identify = attribute[-1].split(' ')[1].replace('{', '').replace('}', '')
                                t_class_identify = class_identify.split(',')[0]
                                f_class_identify = class_identify.split(',')[1]
                                is_first = False
                            line_ = line.split(',')
                            class_TF = line_.pop(-1).replace('\n', '').strip()
                            if class_TF == t_class_identify:
                                class_TF = 1
                            if class_TF == f_class_identify:
                                class_TF = 0
                            line_.append(class_TF)
                            line_ = [float(i) for i in line_]
                            data.append(line_)
                            continue
                data = np.array(data)
                data_file = data[:, 0:-1]
                label_file = data[:, -1]
                if is_normalized:
                    # 数据归一化
                    data_file -= np.array(data_file).min()
                    data_file /= np.array(data_file).max()
                    label_file -= np.array(label_file).min()
                    label_file /= np.array(label_file).max()
                # 加入列表
                data_list.append(data_file)
                label_list.append(label_file)
        # 重采样
        if is_sample:
            for index in range(len(data_list)):
                data_list[index], label_list[index] = MAHAKIL().fit_sample(data_list[index], label_list[index])
        return data_list, label_list

    @staticmethod
    def get_positive_rate(data_list, label_list):
        for index in range(len(data_list)):
            positive = 0
            # 按照正例和反例划分数据集
            N = label_list[index].shape[0]  # 样例总数
            for i in range(N):
                if label_list[index][i] == 1:
                    positive += 1
            print(str(index) + ":positive rate is " + str(positive/N))
