# shot_boundary_detection
# 镜头边界检测
从dy短视频中，选择一个小米发布会宣传视频进行边界检测实验，该短视频中有较多的镜头切换（硬切换）。

实验目的：在不同镜头边界检测方法中，完成对该视频的边界检测并从每一个镜头中抽取一帧作为关键帧保存起来（将切分结果保存为gif形式的动图需要消耗大量时间，且各种方法在时间消耗上无差异），统计完成该任务总的耗时，未对准确率等指标进行测试。  
视频参数：
+ 大小：5.67M
+ 时长：58s
+ 帧率：24
+ 总帧数： 1405   
+ 视频链接：https://github.com/fhlt/shot_boundary_detection/blob/master/test.mp4  

测试环境：
+ 操作系统：64位 windows 10
+ 处理器：i7-8750H CPU@2.20GHz 2.21GHz 
+ GPU：不使用
## method 1
基于窗口最大值和自适应阈值：根据帧图像的灰度值直方图差异进行边缘检测，差异值越大的帧可能就是镜头边缘帧。这种方式可以避免在镜头移动或者图像中出现动态移动的时候差异，提高边缘检测的准确性。其中要注意的地方：
+ 相邻的两个镜头，中间的帧图像个数应该有一个阈值，也就是说帧数相差太少不认可为新的一个镜头。
+ 检测出来的镜头边缘帧，它与前一帧的差值应该是此镜头中，所有帧差中最大的。


+ 代码链接：https://github.com/fhlt/shot_boundary_detection/blob/master/gray_hist.py
+ 耗时：5.02s
## method2
根据相邻帧颜色直方图的bhattacharyya系数进行kmean聚类，以聚类结果作为镜头区分的依据。
+ 代码链接：https://github.com/fhlt/shot_boundary_detection/blob/master/shot_boundary.py
+ 耗时：34.12s

## method3
使用Alexnet深度学习特征，将帧间差异大于某阈值的位置作为镜头分割的依据。（鉴于采用深度学习特征的镜头分割方法太慢了，不再测试deepSDB系列方法）
+ 代码链接：https://github.com/fhlt/shot_boundary_detection/blob/master/Alexnet_SDB.py
+ 耗时：66.66s
