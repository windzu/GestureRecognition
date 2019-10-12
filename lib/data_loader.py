import pandas as pd
import numpy as np

class DataLoader():
    """ 加载csv文件的类
    # Arguments
        path_vid    : 视频的根文件夹的路径
        path_labels : labels路径
        path_train  : 训练集csv路径
        path_val    : 验证集csv路径
        path_test   : 测试集csv路径
    #Returns
        DataLoader实例  
    """
    def __init__(self, path_vid, path_labels, path_train=None, path_val=None, path_test=None):
        self.path_vid    = path_vid
        self.path_labels = path_labels
        self.path_train  = path_train
        self.path_val    = path_val
        self.path_test   = path_test

        self.get_labels(path_labels)

        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)

        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)

        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, mode="input")

    def get_labels(self, path_labels):
        """从csv加载dataframe标签并创建字典，将字符串标签转换为int
        """
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        #提取标签列表
        self.labels = [str(label[0]) for label in self.labels_df.values]
        self.n_labels = len(self.labels)
        #label转为int
        self.label_to_int = dict(zip(self.labels, range(self.n_labels)))
        self.int_to_label = dict(enumerate(self.labels))

    def load_video_labels(self, path_subset, mode="label"):
        """ 从csv加载dataframe
        # Arguments
            path_subset : String, 要加载的csv路径
        #Returns
            A DataFrame
        """
        if mode=="input":
            names=['video_id']
        elif mode=="label":
            names=['video_id', 'label']
        
        df = pd.read_csv(path_subset, sep=';', names=names) 
        
        if mode == "label":
            df = df[df.label.isin(self.labels)]

        return df
    
    def categorical_to_label(self, vector):
        """ 将向量转为相关的字符串标签
        # Arguments
            vector : 代表视频标签的向量
        #Returns
            视频标签
        """
        return self.int_to_label[np.where(vector==1)[0][0]]
