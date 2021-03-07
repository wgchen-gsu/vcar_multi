# Compression Artifact Reduction of Low Bit-rate Videos via Deep Neural Networks using Self-Similarity Prior 

# Dependencies
- Keras, TensorFlow, NumPy, Matlab, OpenCV, ...

# Get Started!
## 1. Preparation
(1)使用编码工具(如HM)对视频进行编码, 编码参数可参见sample_parameters.cfg. 目前的设置: I帧的间隔为32, 即第0, 32, 64帧为I帧, 其余为帧; P帧的编码QP=42, 37, I帧的QP 比P帧的小6~8;
(2)目前只给出了P帧QP = 42 的网络参数, 即models/QP42-HALF-XH-32.06381.hdf5 和 models/QP42-FINAL-28.254367.hdf5, 后续会陆续给出QP=37和QP=32;
(3)目录 video_bits 提供了一些测试视频, 需用  TAppdecoder.exe 解码, 形如: TAppdecoder -b xxx.bin -o xxx.yuv;
(4)Useage: python demo.py --video_coded your_decoded_video.yuv --video_ori your_ori_uncompressed.yuv --QP 42 --height 480 --width 832
Note: 请按你的实际视频修改 height 和 width参数
(5)输出缺省将保存在 output_A 目录, 一帧对应一幅输出图像, 从左到右依次为未编码的原始图像, 解码图像, 经Compression Artifact reduction 处理的图像;
(6)可通过提供的 comp_psnr_3im.m 计算PSNR和SSIM的增益.

# TODOs
QP=37, 32的模型参数; 用于训练的代码.