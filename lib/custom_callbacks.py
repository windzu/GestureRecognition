from keras.callbacks import Callback

import matplotlib as mpl
#mpl.use('TkAgg')  
import matplotlib.pyplot as plt

import json


class HistoryGraph(Callback):
    """用于callback.
    """
    def __init__(self, model_path_name):
        self.model_path_name = model_path_name

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        self.save_training_history(self.model_path_name, self.history)

    def save_training_history(self, path, history):
        """保存历史记录及创建损失图
        # Arguments
            path    : 保存路径
            history : 历史模型
        """
        for metric in history:
            if "val" not in metric:
                plt.clf()

                # 绘制loss损失
                history[metric]=list(map(float, history[metric]))
                plt.plot(history[metric])
                plt.plot(history["val_" + metric])
                plt.title('model ' + metric)
                plt.ylabel(metric)
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.gcf().savefig(path + '/'+metric+'_history' + '.jpg')


        with open(path + '/log' + '.json', 'w') as fp:
            json.dump(history, fp, indent=True)
