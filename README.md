# MobileNetV2 with dynamically generated Feature Pyramid Network (FPN)

This project combines the MobileNetV2 architecture of Inverted Residual Layers with a dynamically generated Feature Pyramid Network. The network outputs an array of feature maps with high semantic meaning in different resolutions. The number of feature maps depends on the number of Inverted Residual Blocks which reduce the resolution (stride: 2). They can be specified in `self.inverted_residual_setting`.

### Implementation hurdles

When implementing this code I had to solve a number of different problems.
First of all, as this was my first PyTorch project, I had to understand the syntax, get to know the modules and layers as well as the functions availiable in PyTorch. In addition to that I read the papers of the Feature Pyramid and MobileNetV2 network to understand how I could combine the best of both of them in a meaningful way.

To further deepen my understanding of how the ideas of the papers were translated into the starter code I altered every network and compared the output of different settings. I also had a look (printed) at a lot of the modules and tensors in this process.

Once I understood how the two code bases work I started to combine them into a single file. I quickly noticed that my code had a lot of repetitions, especially when defining the layers or connecting them. I remembered that somewhere along the way of learning PyTorch I read that the beauty of this framework was that it was completely dynamical and one could use almost any python logic. So I looped away! My code was amazingly pythonic, but when I printed out my network all my layers were gone. :(
Long story short: `nn.ModuleList` saved me.

I also changed the `self.inverted_residual_setting` to a dicionary to improve the readability of my code.

Once I had all my layers done it was time to define my `forward` function. After a lot of thinking and googling of how I could possibly connect my nameless layers of differnt quantities in this not straight forward way that is need, the solution was suprisingly realtivly simple. Some more loops and status tracking (`n_lateral_connections`) did the job perfectly.

### Starter code:

- Pytorch base for MobileNetV2: https://github.com/tonylins/pytorch-mobilenet-v2

- Pytorch base for FPN and object detection: https://github.com/kuangliu/torchcv/tree/master/torchcv/models/fpnssd

### Other sources:

- Feature Pyramid Networks for Object Detection https://arxiv.org/pdf/1612.03144.pdf

- MobileNetV2: Inverted Residuals and Linear Bottlenecks https://arxiv.org/pdf/1801.04381.pdf

- Understanding Feature Pyramid Networks for object detection (FPN):
  https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c

- PyTorch Docs:
  https://pytorch.org/docs/stable/index.html

- When should I use nn.ModuleList and when should I use nn.Sequential? https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
