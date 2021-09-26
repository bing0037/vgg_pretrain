import numpy as np
import torch
import math
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


# define customized functions with customized gradients
class STEFunction(torch.autograd.Function):
    """ define straight through estimator with overrided gradient (gate) """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return torch.mul(F.softplus(input), grad_output)
#         return grad_output
 

    
# define customized layers for pruning
class LinearLayerW(nn.Module):
    """ Custom Linear layer Weight Pruning """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(torch.Tensor(size_out, size_in))  # nn.Parameter is a Tensor that's a module parameter.
        self.weight_aux = nn.Parameter(torch.Tensor(size_out, size_in))  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = nn.Parameter(torch.Tensor(size_out))

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        nn.init.uniform_(self.weight_aux) # weight init for aux parameter, can be truncated normal
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        weight_mask = STEFunction.apply(self.weight_aux) # 0-1 matrix
        weight_sparse = torch.mul(self.weight, weight_mask)
        w_times_x = torch.mm(x, weight_sparse.t())
        w_times_x_plus_b = torch.add(w_times_x, self.bias)  # w times x + b
        return w_times_x_plus_b
    
class LinearLayerN(nn.Module):
    """ Custom Linear layer Neuron Pruning """
    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in, self.size_out = size_in, size_out
        weight = torch.Tensor(size_out, size_in)
        self.weight = nn.Parameter(torch.Tensor(size_out, size_in))
        self.weight_aux = nn.Parameter(torch.Tensor(size_out))
        self.bias = nn.Parameter(torch.Tensor(size_out))

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # weight init
        nn.init.uniform_(self.weight_aux) # weight init for aux parameter, can be truncated normal
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init
    
        
    def forward(self, x):
        w_times_x = torch.mm(x, self.weight.t())
        w_times_x_plus_b = torch.add(w_times_x, self.bias)
        neuron_mask = STEFunction.apply(self.weight_aux)
        neuron_sparse = torch.mul(w_times_x_plus_b, neuron_mask)
        return neuron_sparse
    
class Conv2DLayerW(nn.Module):
    """ Custom Conv2D layer Weight Pruning """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, stride=1, dilation=1, groups=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_aux = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.weight_aux) # weight init for aux parameter, can be truncated normal
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        weight_mask = STEFunction.apply(self.weight_aux)
        weight_sparse = torch.mul(self.weight, weight_mask)
        output = F.conv2d(x, weight_sparse, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return output
    
class Conv2DLayerN(nn.Module):
    """ Custom Conv2D layer Neuron Pruning """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, stride=1, dilation=1, groups=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_aux = nn.Parameter(torch.Tensor(out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.weight_aux) # weight init for aux parameter, can be truncated normal
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        neuron_mask = STEFunction.apply(self.weight_aux)
#         weight_sparse = torch.mul(self.weight, neuron_mask[:,None,None,None]) # expend dimension 
#         output = F.conv2d(x, weight_sparse, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        output = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        output = torch.mul(output, neuron_mask[:,None,None]) 
        return output
    
# build CNN backbone and feedforward classifier
def make_features(cfg, pruning=None, batch_norm=False): 
    """
       pruning options: 
       W:weight pruning, N:neuron pruning, None:not pruning
       Leave some room for more generalized/customized pruning 
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if pruning == "W":
                conv2d = Conv2DLayerW(in_channels, v, kernel_size=3, padding=1)
            elif pruning == "N":
                conv2d = Conv2DLayerN(in_channels, v, kernel_size=3, padding=1)
            elif not pruning:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_classifier(num_classes=1000, pruning=None, Dropout_p=0.5): 
    """ 
        num_classes options: 10:CIFAR10 OR 1000:IMAGENET(default) or others 
        pruning options: W:weight pruning, N:neuron pruning, None:not pruning
    """
    layers = []
    if pruning == "W":
        if num_classes == 10:
            classifier = nn.Sequential(
                LinearLayerW(512, 512),
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                LinearLayerW(512, num_classes),
            )
        else:
            classifier = nn.Sequential(
                LinearLayerW(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                LinearLayerW(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                LinearLayerW(4096, num_classes),
            )
    elif pruning == "N":
        if num_classes == 10:
            classifier = nn.Sequential(
                LinearLayerN(512, 512), 
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                LinearLayerN(512, num_classes),
            )
        else:
            classifier = nn.Sequential(
                LinearLayerN(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                LinearLayerN(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                LinearLayerN(4096, num_classes),
            )
    else:
        if num_classes == 10:
            classifier = nn.Sequential(
                nn.Linear(512, 512), 
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                nn.Linear(512, num_classes),
            )
        else:
            classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(Dropout_p),
                nn.Linear(4096, num_classes),
            )    
    return classifier

# Build model and load/initialize weights
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """ 
        function to help weight initialization
        Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class VGG(nn.Module):

    def __init__(self, features, classifier, num_classes, model_weights):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = classifier
        self.num_classes = num_classes
        if model_weights:
            print("Load pretrained model weights.")
            self._load_weights(model_weights)
        else:
            print("Initialize model weights with default weight distribution.")
            self._initialize_weights()
        self.num_gates = self._get_num_gates()
        

    def forward(self, x):
        x = self.features(x)
#         print("x shape before flattening:", x.shape)
        x = x.view(x.size(0), -1)
#         print("x shape after flattening:", x.shape)
        x = self.classifier(x)
#         sparsity = torch.div(self._get_num_opened_gates(), self.num_gates)
        return x#, sparsity
    
    def _get_num_gates(self):
        num_gates = torch.tensor(0.)
        for m in self.modules():
            if isinstance(m, (Conv2DLayerW, Conv2DLayerN, LinearLayerW, LinearLayerN)):
                num_gates += m.weight_aux.numel()
        return num_gates
    
#     def _get_num_opened_gates(self):
#         num_opened_gates = torch.tensor(0.)
#         for m in self.modules():
#             if isinstance(m, (Conv2DLayerW, Conv2DLayerN, LinearLayerW, LinearLayerN)):
#                 num_opened_gates += torch.sum(m.mask)
#         return num_opened_gates

    def _initialize_weights(self): 
        """ weight ini has been implemented in customized layers, but this one can be used for global change """
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2DLayerW) or isinstance(m, Conv2DLayerN):
#                 print(i, "init_conv")
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if not isinstance(m, nn.Conv2d):
                    _no_grad_trunc_normal_(m.weight_aux, mean=0.2, std=0.1, a=0.1, b=0.3)
            elif isinstance(m, nn.BatchNorm2d):
#                 print(i, "init_bn")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) or isinstance(m, LinearLayerW) or isinstance(m, LinearLayerN):
#                 print(i, "init_linear")
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                if not isinstance(m, nn.Linear):
                    _no_grad_trunc_normal_(m.weight_aux, mean=0.2, std=0.1, a=0.1, b=0.3)
                
    def _load_weights(self, model_weights):
        """ loading pretrained weights"""
        i = 0
        for m in self.modules():
            if not isinstance(m, (nn.Conv2d, Conv2DLayerW, Conv2DLayerN, nn.BatchNorm2d, nn.Linear, \
                                  LinearLayerW, LinearLayerN, nn.ReLU, nn.Dropout, nn.MaxPool2d)):
                continue
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2DLayerW) or isinstance(m, Conv2DLayerN):
#                 m.weight.copy_(model_weights['features.' + str(i) + '.weight'])
                #or
#                 print(i, "init_conv")
                m.weight.data = model_weights['features.module.' + str(i) + '.weight']
                if m.bias is not None:
                    m.bias.data = model_weights['features.module.' + str(i) + '.bias']
                if not isinstance(m, nn.Conv2d):
#                     nn.init.trunc_normal_(m.weight_aux, mean=0.5, std=0.1, a=0.3, b=0.7)#this initialization approach was introduced in later versions of pytorch
                    _no_grad_trunc_normal_(m.weight_aux, mean=0.2, std=0.1, a=0.1, b=0.3)
            elif isinstance(m, nn.BatchNorm2d):
#                 print(i, "init_bn")
                m.weight.data = model_weights['features.module.' + str(i) + '.weight']
                m.bias.data = model_weights['features.module.' + str(i) + '.bias']
                m.running_mean.data = model_weights['features.module.' + str(i) + '.running_mean']
                m.running_var.data = model_weights['features.module.' + str(i) + '.running_var']
            elif isinstance(m, nn.Linear) or isinstance(m, LinearLayerW) or isinstance(m, LinearLayerN):
#                 print(i, "init_linear")
                m.weight.data = model_weights['classifier.' + str(i) + '.weight']
                m.bias.data = model_weights['classifier.' + str(i) + '.bias']
                if not isinstance(m, nn.Linear):
#                     nn.init.trunc_normal_(m.weight_aux, mean=0.5, std=0.1, a=0.3, b=0.7)#this initialization approach was introduced in later versions of pytorch
                    _no_grad_trunc_normal_(m.weight_aux, mean=0.2, std=0.1, a=0.1, b=0.3)
            i += 1
            if i >= len(self.features):
                i = 0

                
    def train_fz_bn(self, freeze_bn=True, freeze_bn_affine=True, mode=True):
        """
            Override the default train() to freeze the BN parameters
        """
        super(VGG, self).train(mode)
#         if freeze_bn:
# #             print("Freezing Mean/Var of BatchNorm2D.")
#             if freeze_bn_affine:
# #                 print("Freezing Weight/Bias of BatchNorm2D.")
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                        
cfg = {
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def vgg16(pruning=None, num_classes=10, pretrain=True, Dropout_p=0.5):
    """VGG 16-layer model
    Args:
        pruning (str):      None, ordinary model without pruning
                            W, weight pruning model
                            N, neuron pruning model (kernal pruning for CNNs)
                        
        num_classes (int):  10, Cifar10 or models with 10 classes
                            1000, ImageNet or models with 1000 classes
                            
        pretrain (bool):    True, load pretrain weights
                            False, initialize weights and train from scratch

        Dropout_p:          Dropout probability of an element to be zeroed. Default: 0.5

    """
    print("Dropout probability:", Dropout_p)
    if pruning == "W":
        print("Construct autoprune weight pruning module")
    elif pruning == "N":
        print("Construct autoprune neuron pruning module")
    elif not pruning:
        print("Construct ordinary model without pruning")
    else:
        print(pruning, "pruning indicator is not recognized, construct ordinary model")
        pruning = None
        
    features = make_features(cfg['VGG16'], pruning=pruning, batch_norm=True)
    classifier = make_classifier(num_classes=num_classes, pruning=pruning, Dropout_p=Dropout_p)
    
    if pretrain:
        if num_classes == 1000:
            vgg16_bn = torch.load('vgg16_bn-6c64b313.pth')
        elif num_classes == 10:
            vgg16_bn = torch.load('vgg16_bn_cifar10.pth')
    else:
        vgg16_bn = None
        
    model = VGG(features, classifier, num_classes, model_weights=vgg16_bn)

    return model
