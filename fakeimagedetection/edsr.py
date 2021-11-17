# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.utils.model_zoo
import ssl



url = {
    "r16f64x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt",
    "r16f64x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt",
    "r16f64x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt",
    "r32f256x2": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt",
    "r32f256x3": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt",
    "r32f256x4": "https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt",
}


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def make_model(args, parent=False):
    return EDSR(args)


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()

        n_resblocks = args["n_resblocks"]
        n_feats = args["n_feats"]
        kernel_size = 3
        scale = args["scale"][0]
        act = nn.ReLU(True)
        url_name = "r{}f{}x{}".format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        self.sub_mean = MeanShift(args["rgb_range"])
        self.add_mean = MeanShift(args["rgb_range"], sign=1)

        # define head module
        m_head = [conv(args["n_colors"], n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=args["res_scale"])
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args["n_colors"], kernel_size),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # =============================================================================
        #         print('\nHere\n')
        #         print(x.shape)
        #         print(x[0,0,0:10])
        # =============================================================================
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))


def load(model, pre_train="download", resume=0, cpu=False):
    load_from = None
    kwargs = {}
    ssl._create_default_https_context = ssl._create_unverified_context
    if pre_train == "download":
        # print("Downloading the model")
        # dir_model = os.path.join('..', 'models')
        # os.makedirs(dir_model, exist_ok=True)
        load_from = torch.utils.model_zoo.load_url(
            model.url,
            # model_dir=dir_model,
            **kwargs
        )
    elif pre_train:
        # print("Loading the model from {}".format(pre_train))
        load_from = torch.load(pre_train, **kwargs)
    if load_from:
        model.load_state_dict(load_from, strict=False)
        
def load_edsr(device, n_resblocks=16, n_feats=64, scale=4, model_details=True):
    """
    Loads the EDSR model

    Parameters
    ----------
    device : str
        device type.
    n_resblocks : int, optional
        number of res_blocks. The default is 16.
    n_feats : int, optional
        number of features. The default is 64.

    Returns
    -------
    model : torch.nn.model
        EDSR model.

    """
    args = {
        "G0": 64,
        "RDNconfig": "B",
        "RDNkSize": 3,
        "act": "relu",
        "batch_size": 16,
        "betas": (0.9, 0.999),
        "chop": True,
        "cpu": True,
        "data_range": "1-800/801-810",
        "data_test": ["Demo"],
        "data_train": ["DIV2K"],
        "debug": False,
        "decay": "200",
        "dilation": False,
        "dir_data": "../../../dataset",
        "dir_demo": "../test",
        "epochs": 300,
        "epsilon": 1e-08,
        "ext": "sep",
        "extend": ".",
        "gamma": 0.5,
        "gan_k": 1,
        "gclip": 0,
        "load": "",
        "loss": "1*L1",
        "lr": 0.0001,
        "model": "EDSR",
        "momentum": 0.9,
        "n_GPUs": 1,
        "n_colors": 3,
        "n_feats": 64,
        "n_resblocks": 16,
        "n_resgroups": 10,
        "n_threads": 6,
        "no_augment": False,
        "optimizer": "ADAM",
        "patch_size": 192,
        "pre_train": "download",
        "precision": "single",
        "print_every": 100,
        "reduction": 16,
        "res_scale": 1,
        "reset": False,
        "resume": 0,
        "rgb_range": 255,
        "save": "test",
        "save_gt": False,
        "save_models": False,
        "save_results": True,
        "scale": [scale],
        "seed": 1,
        "self_ensemble": False,
        "shift_mean": True,
        "skip_threshold": 100000000.0,
        "split_batch": 1,
        "template": ".",
        "test_every": 1000,
        "test_only": True,
        "weight_decay": 0,
    }
    model = make_model(args).to(device)
    load(model)
    return model