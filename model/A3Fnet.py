import warnings
warnings.filterwarnings("ignore")
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from model.box_filter import BoxFilter
from torch.autograd import Variable

class Down(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.bn(x))

class Image_Prediction_Generator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x):
        gt_pre = self.conv(x)
        return x, gt_pre

class A3Fnet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[32, 64, 128, 256, 512]):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
        )
        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
        )

        self.Down1 = Down(c_list[0])
        self.Down2 = Down(c_list[1])
        self.Down3 = Down(c_list[2])
        self.Down4 = Down(c_list[3])

        self.decoder1 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[4], 3, stride=1, padding=1),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], 3, stride=1, padding=1),
            nn.Conv2d(c_list[3], c_list[3], 3, stride=1, padding=1),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[2], 3, stride=1, padding=1),
            nn.Conv2d(c_list[2], c_list[2], 3, stride=1, padding=1),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
            nn.Conv2d(c_list[1], c_list[1], 3, stride=1, padding=1),
        )

        self.pred4 = Image_Prediction_Generator(c_list[3])
        self.pred3 = Image_Prediction_Generator(c_list[2])
        self.pred2 = Image_Prediction_Generator(c_list[1])
        self.pred1 = Image_Prediction_Generator(c_list[0])

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.dbn0 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[3])
        self.dbn2 = nn.GroupNorm(4, c_list[2])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])


        self.final = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
            nn.Conv2d(c_list[0], c_list[0], 3, stride=1, padding=1),
            nn.Conv2d(c_list[0], num_classes, kernel_size=1),
        )

        self.apply(self._init_weights)

        self.gf = GuidedFilter_attention(r=2, eps=1e-2)

        self.attentionblock4 = AttentionBlock(in_channels=256)
        self.attentionblock3 = AttentionBlock(in_channels=128)
        self.attentionblock2 = AttentionBlock(in_channels=64)
        self.attentionblock1 = AttentionBlock(in_channels=32)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        _, _, img_shape, _ = x.size()
        out = self.encoder1(x)
        t1 = out
        out = F.gelu(self.Down1(self.ebn1(out)))


        out = self.encoder2(out)
        t2 = out
        out = F.gelu(self.Down2(self.ebn2(out)))


        out = self.encoder3(out)
        t3 = out
        out = F.gelu(self.Down3(self.ebn3(out)))


        out = self.encoder4(out)
        t4 = out
        out = F.gelu(self.Down4(self.ebn4(out)))


        out = self.encoder5(out)
        out = F.gelu(out)

        ###############################################################################################
        out = self.decoder1(out)
        out = F.gelu(self.dbn0(out))

        out4 = out
        out, gt_pre4 = self.pred4(out)
        gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)

        ##
        N, C, H, W = t4.size()
        t4_small = F.upsample(t4, size=(int(H/2), int(W/2)),mode='bilinear')
        out = self.gf(t4_small,out4 , t4, self.attentionblock4(t4_small, out4))

        out = F.gelu(F.interpolate(self.dbn1(out), scale_factor=(2, 2), mode='bilinear',
                                   align_corners=True))
        out = self.decoder2(out)


        out3 = out
        out, gt_pre3= self.pred3(out)
        gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)

        ##
        N, C, H, W = t3.size()
        t3_small = F.upsample(t3, size=(int(H/2), int(W/2)),mode='bilinear')
        out = self.gf(t3_small, out3, t3, self.attentionblock3(t3_small, out3))

        out = F.gelu(F.interpolate(self.dbn2(out), scale_factor=(2, 2), mode='bilinear',
                                   align_corners=True))
        out = self.decoder3(out)


        out2 = out
        out, gt_pre2 = self.pred2(out)
        gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)

        ##
        N, C, H, W = t2.size()
        t2_small = F.upsample(t2, size=(int(H / 2), int(W / 2)), mode='bilinear')
        out = self.gf(t2_small, out2, t2, self.attentionblock2(t2_small, out2))

        out = F.gelu(F.interpolate(self.dbn3(out), scale_factor=(2, 2), mode='bilinear',
                                   align_corners=True))
        out = self.decoder4(out)


        out1 = out
        out, gt_pre1 = self.pred1(out)
        gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)

        ##
        N, C, H, W = t1.size()
        t1_small = F.upsample(t1, size=(int(H / 2), int(W / 2)), mode='bilinear')
        out = self.gf(t1_small, out1, t1, self.attentionblock1(t1_small, out1))

        out = F.interpolate(self.dbn4(out), scale_factor=(2, 2), mode='bilinear', align_corners=True)  # b, num_class, H, W
        out = self.final(out)


        gt_pre1 = torch.sigmoid(gt_pre1)
        gt_pre2 = torch.sigmoid(gt_pre2)
        gt_pre3 = torch.sigmoid(gt_pre3)
        gt_pre4 = torch.sigmoid(gt_pre4)

        return (gt_pre4, gt_pre3, gt_pre2, gt_pre1), torch.sigmoid(out)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,kernel_size=1)

        self.phi = nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels,kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode='bilinear')
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = F.sigmoid(self.psi(f))

        return sigm_psi_f


class GuidedFilter_attention(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter_attention, self).__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)
        self.epss = 1e-12

    def forward(self, lr_x, lr_y, hr_x, l_a):
        n_lrx, c_lrx, h_lrx, w_lrx = lr_x.size()
        n_lry, c_lry, h_lry, w_lry = lr_y.size()
        n_hrx, c_hrx, h_hrx, w_hrx = hr_x.size()

        lr_x = lr_x.double()
        lr_y = lr_y.double()
        hr_x = hr_x.double()
        l_a = l_a.double()

        assert n_lrx == n_lry and n_lry == n_hrx
        assert c_lrx == c_hrx and (c_lrx == 1 or c_lrx == c_lry)
        assert h_lrx == h_lry and w_lrx == w_lry
        assert h_lrx > 2*self.r+1 and w_lrx > 2*self.r+1

        N = self.boxfilter(Variable(lr_x.data.new().resize_((1, 1, h_lrx, w_lrx)).fill_(1.0)))

        l_a = torch.abs(l_a) + self.epss

        t_all = torch.sum(l_a)
        l_t = l_a / t_all

        mean_a = self.boxfilter(l_a) / N
        mean_a2xy = self.boxfilter(l_a * l_a * lr_x * lr_y) / N
        mean_tax = self.boxfilter(l_t * l_a * lr_x) / N
        mean_ay = self.boxfilter(l_a * lr_y) / N
        mean_a2x2 = self.boxfilter(l_a * l_a * lr_x * lr_x) / N
        mean_ax = self.boxfilter(l_a * lr_x) / N

        temp = torch.abs(mean_a2x2 - N * mean_tax * mean_ax)
        A = (mean_a2xy - N * mean_tax * mean_ay) / (temp + self.eps)
        b = (mean_ay - A * mean_ax) / (mean_a)

        A = self.boxfilter(A) / N
        b = self.boxfilter(b) / N

        mean_A = F.upsample(A, (h_hrx, w_hrx), mode='bilinear')
        mean_b = F.upsample(b, (h_hrx, w_hrx), mode='bilinear')

        return (mean_A*hr_x+mean_b).float()


