import torch
import torch.nn as nn
from torchvision import models
from einops.layers.torch import Rearrange


class ROI_Pool(nn.Module):

    def __init__(self, size):
        super(ROI_Pool, self).__init__()
        assert len(size) == 2, 'size参数输入(长, 宽)'

        self.roi_pool = nn.AdaptiveMaxPool2d(size)

    def forward(self, feature_maps):
        assert feature_maps.dim() == 4, '应有尺寸为 (N, C, H, W)'
        return self.roi_pool(feature_maps)


class VGG(nn.Module):
    def __init__(self, num_classes=20) -> None:
        super().__init__()

        # VGG16模型的卷积层设置，取消最后一个最大池化层'M'
        feature_list = [64, 64, 'M', 128, 128, 'M', 256,
                        256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]

        self.model = models.vgg.make_layers(feature_list)

        # ROI池化层
        self.roi_pool = ROI_Pool((7, 7))

        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )

        self.softmax = nn.Linear(4096, num_classes + 1)

        self.bbox = nn.Linear(4096, num_classes * 4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.roipool(x)
        x = self.classifier(x)
        classify = self.softmax(x)
        regression = self.bbox(x)
        return classify, regression
    

if __name__ == '__main__':
    # model = models.vgg16(pretrained=True)
    model = VGG(3)
    print(model)
