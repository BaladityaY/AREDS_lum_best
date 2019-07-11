import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class GaussianLayer(nn.Module): #(nn.Conv2d): #(nn.Module):
    def __init__(self, chans, kernel=21, stride=1):
        self.kernel = kernel
        self.chans = chans
        self.groups = chans
        self.stride = stride
        self.padding = 10
        self.dilation = 1
        
        
        #super(GaussianLayer, self).__init__(self.chans, self.chans, self.kernel, stride=self.stride,
        #                                    padding=0, bias=None, groups=self.chans)
        
        super(GaussianLayer, self).__init__()
        
        self.sigma = torch.nn.Parameter(torch.randn(1, requires_grad=True))
        
        self.x = torch.nn.Parameter(torch.arange(0, self.kernel, requires_grad=False).type(torch.FloatTensor))
        self.y = torch.nn.Parameter(torch.arange(0, self.kernel, requires_grad=False).type(torch.FloatTensor))
        
        #self.weight = torch.nn.Parameter(self.create_kernel().repeat(self.chans,1,1,1))
        self.bias = torch.nn.Parameter(torch.Tensor(self.chans))
        
        #self.weight = torch.nn.Parameter(self.create_kernel().repeat(self.chans,1,1,1))
        
        #self.gauss_conv = nn.Conv2d(self.chans, self.chans, self.kernel, stride=self.stride,
        #                            padding=0, bias=None, groups=self.chans)
        
        #self.weights_init()
        
    #'''    
    def forward(self, x):
        self.weight = self.create_kernel().repeat(self.chans,1,1,1)
        
        #print(self.weight.shape)
        #print(self.weight)
        
        #return super(GaussianLayer, self).forward(x)
    
        return self.sigma, F.conv2d(x, self.weight, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
        #return self.gauss_conv(x)
    #'''
    
    def create_kernel(self): #weights_init(self):
        mean = [0, 0]

        XX, YY = torch.meshgrid((self.x, self.y))
        XX = XX.transpose(1,0)
        YY = YY.transpose(1,0)

        XX = XX - int(len(self.x)/2)
        YY = int(len(self.y)/2) - YY
        
        xx = (XX - mean[0])
        yy = (YY - mean[1])
        
        #clamp_min = 0
        clamp_max = 20

        sig_clamp = torch.clamp(self.sigma, -clamp_max, clamp_max)

        gauss_kernel = torch.exp(-1*((xx*xx + yy*yy)/(2*(sig_clamp**2))))
        #guass_kernel = #scipy.ndimage.gaussian_filter(n, sigma=3)
        
        return gauss_kernel
        
        '''
        for name, f in self.named_parameters():
            if name == 'gauss_conv.weight':
                print('name: {}'.format(name))
                f.data.copy_(gauss_kernel)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.data.copy_(guass_kernel)
        '''

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=16, zero_init_residual=False): #changed num_classes from 1000 to 4
        super(ResNet, self).__init__()
        self.inchans = 1
        self.inplanes = 64
        
        self.gauss_conv = GaussianLayer(self.inchans)
        self.conv1 = nn.Conv2d(self.inchans, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        sig, new_x = self.gauss_conv(x)
        print('sigma: {}'.format(sig))
        x = x - new_x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #print('preavg: {}'.format(x.shape))
        x = self.avgpool(x)
        #print('postavg: {}'.format(x.shape))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
