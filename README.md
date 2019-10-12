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
