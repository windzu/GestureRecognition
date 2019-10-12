# 基于jester数据集的手势识别项目
3D版本ResNet101，基于keras+tensorflow+python3


## 使用说明
```
python main.py --config <configuration_file>
```
Example:
```
# 训练/测试
python main.py --config "config.cfg"

# 摄像头实时测试
python camer_test.py --config "config.cfg"

# 预拍摄视频测试
python single_video_test.py --config "config.cfg"
```
## 模型及测试视频
模型和测试视频很大，放在了坚果云的共享文件中(注册才可以下载)
下载以下链接中的文件，解压放在根目录即可
[测试视频](https://www.jianguoyun.com/p/DXdEQFgQovDTBhjv3_4B)
[model](https://www.jianguoyun.com/p/Db_23N4QovDTBhjF3_4B)



## 数据集为20BN-Jester
[数据集地址](https://20bn.com/datasets/jester)
共包含27个类别
