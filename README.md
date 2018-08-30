# MobileNetV2 with dynamically generated Feature Pyramid Network (FPN)

This project combines the MobileNetV2 architecture of Inverted Residual Layers with a dynamically generated Feature Pyramid Network. The network outputs an array of feature maps with high semantic meaning in different resolutions. The number of feature maps depends on the number of Inverted Residual Blocks which reduce the resolution (stride: 2). They can be specified in `self.inverted_residual_setting`.

### Starter code:

- Pytorch base for MobileNetV2: https://github.com/tonylins/pytorch-mobilenet-v2

- Pytorch base for FPN and object detection: https://github.com/kuangliu/torchcv/tree/master/torchcv/models/fpnssd

### Other sources:

- Feature Pyramid Networks for Object Detection https://arxiv.org/pdf/1612.03144.pdf

- MobileNetV2: Inverted Residuals and Linear Bottlenecks https://arxiv.org/pdf/1801.04381.pdf

- Understanding Feature Pyramid Networks for object detection (FPN):
  https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
