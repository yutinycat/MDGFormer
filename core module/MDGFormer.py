import torch
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from Res2Net_v1b import res2net50_v1b_26w_4s
from TSBD import TSBD

def weight_init(module):
    for n, m in module.named_children():
      #  print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential,nn.ModuleList,nn.ModuleDict)):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (LayerNorm,nn.ReLU,Act,nn.AdaptiveAvgPool2d,nn.Softmax,nn.AvgPool2d)):
            pass
        else:
            m.initialize()
            
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x
    
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel=64):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat([x0, x1, x2, x3], dim=1))
        x = x_cat + self.conv_res(x)
        return x
    
    def initialize(self):
        weight_init(self)
        
class Fre_D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Fre_D, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.convH = BasicConv2d(in_ch, out_ch)
        self.convL = BasicConv2d(3*in_ch, out_ch)
        
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:,:,0,::]
        y_LH = yH[0][:,:,1,::]
        y_HH = yH[0][:,:,2,::]
        yH = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yL = self.convH(yL)
        yH = self.convL(yH)
        return yL, yH #低频，高频

class LLFE(nn.Module): ###lowl level feature encoder
    def __init__(self, channel=64, heads=2):
        super(LLFE, self).__init__()
        self.convH = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.convL = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv1 = BasicConv2d(channel, channel, kernel_size=1)
        self.heads = heads
        self.p = nn.Parameter(torch.ones((1),requires_grad=True))
        
    def forward(self, low, high):
        b,c,h,w = low.shape
        scale = c ** -0.5
        q_h = self.convH(high)
        q_h = rearrange(q_h, 'b (head c) h w -> b head (h w) c', head=self.heads)
        
        v_h = self.convH(high)
        
        k_l = self.convL(low)
        k_l = rearrange(k_l, 'b (head c) h w -> b head (h w) c', head=self.heads)
        
        v_l = self.convL(low)
        v_l = rearrange(v_l, 'b (head c) h w -> b head (h w) c', head=self.heads)
        
        attn_h = q_h @ k_l.transpose(-2, -1) * scale * self.p
        attn_h = attn_h.softmax(dim=-1)

        out = attn_h @ v_l
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.heads, h=h) + v_h
        return out
    
    def initialize(self):
        weight_init(self)

class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h
    
class FRGC(nn.Module): ###融合频率以及RGB分量
    def __init__(self, channel=64, p_mid=16, mids=4):
        super(FRGC, self).__init__()
        self.num_s = int(p_mid)
        self.num_n = (mids) * (mids)
        self.pr = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv1 = BasicConv2d(channel, p_mid, kernel_size=3, padding=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n) #16 16
        self.conv2 = BasicConv2d(p_mid, channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(2*channel, channel, kernel_size=3, padding=1)
        
        # self.p = nn.Parameter(torch.ones((channel,1,1),requires_grad=True))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, freq, rgb):
        b, c, h, w = rgb.shape
        rgb2 = self.conv1(rgb) # b 16 h w
        Fr = self.pr(rgb2)[:, :, 1:-1, 1:-1].reshape(b, self.num_s, -1) ### b 16 16
        Fgc = self.gcn(Fr)
        Frgb = rgb2.reshape(b, self.num_s, -1) #b 16 hw
        F1 = rgb + self.conv2(torch.matmul(Fgc, Frgb).reshape(b, self.num_s, h, w)) ##RGB
        # W = self.sigmoid(F1*self.p)
        W = self.sigmoid(F1)
        
        out1 = self.relu(self.conv3(freq*W)+self.conv3(freq*(1-W)))
        out2 = self.conv4(torch.cat([out1, rgb], dim=1))
        
        # out2 = self.conv4(torch.cat([out1, rgb], dim=1))+freq
        return out2
    
class HLFE(nn.Module): ###high level feature encoder
    def __init__(self, channel=64, heads=2):
        super(HLFE, self).__init__()
        self.convH = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.convL = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv1 = BasicConv2d(channel, channel, kernel_size=1)
        self.conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(2*channel, channel, kernel_size=3, padding=1)
        self.heads = heads
        self.sigmoid = nn.Sigmoid()
        self.fgrc = FRGC(channel)
        self.p = nn.Parameter(torch.ones((1),requires_grad=True))
        
    def forward(self, low, high, rgb):
        b,c,h,w = low.shape
        scale = c ** -0.5
        q_h = self.convH(high)
        q_h = rearrange(q_h, 'b (head c) h w -> b head (h w) c', head=self.heads)
        
        v_h = self.convH(high)
        
        k_l = self.convL(low)
        k_l = rearrange(k_l, 'b (head c) h w -> b head (h w) c', head=self.heads)
        
        v_l = self.convL(low)
        v_l = rearrange(v_l, 'b (head c) h w -> b head (h w) c', head=self.heads)
        
        attn_h = q_h @ k_l.transpose(-2, -1) * scale * self.p
        attn_h = attn_h.softmax(dim=-1)

        out = attn_h @ v_l
        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.heads, h=h) + v_h

        h,w = rgb.size()[-2:]
        F1 = F.interpolate(out,size=(h,w), mode='bicubic', align_corners=False)
        F2 = self.conv2(rgb)
        out_2 = self.fgrc(F1, F2)
                         
        return out_2
    
    def initialize(self):
        weight_init(self)    

class WA(nn.Module): ###  Weight Allocation
    def __init__(self, channel=64):
        super(WA, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(2*channel, channel, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.Av = nn.AdaptiveAvgPool2d(1)
        self.Mx = nn.AdaptiveMaxPool2d(1)
        
    def forward(self, x1, x2):
        W = self.sigmoid(self.conv3(torch.cat([x1, x2], dim=1)))
        out = x1+x2+x1*W+x2*(1-W)
        Wl = self.sigmoid(self.conv3(torch.cat([self.Av(out),self.Mx(out)],dim=1)))
        
        return Wl

class FFusion(nn.Module):
    def __init__(self, channel=64):
        super(FFusion, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(2*channel, channel)
        self.conv4 = BasicConv2d(channel, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.wa = WA()
        
    def forward(self, x1, x2): ###X1:up X2:down
        F1_1 = self.conv2(x1)
        F2_1 = self.conv2(x2)
        Wl = self.wa(F1_1, F2_1)
        
        F1_2 = self.conv4(F1_1)
        F2_2 = self.conv4(F2_1)
        
        out = (F1_2*F2_1+F2_1)*Wl+(F2_2*F1_1+F1_1)*(1-Wl)
        out_f = self.conv3(torch.cat([out,F2_1], dim=1))
        return out_f
    
class Cat(nn.Module):
    def __init__(self, channel=64):
        super(Cat, self).__init__()
        self.conv1 = BasicConv2d(channel, channel)
        self.conv2 = BasicConv2d(2*channel, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x1, x2): ##x1上面 x2下面
        h1,w1 = x1.size()[-2:]
        x2 = F.interpolate(x2,size=(h1,w1), mode='bicubic', align_corners=False)
        F1 = self.conv1(x1)
        F2 = self.conv1(x2)
        Ws = self.sigmoid(self.conv2(torch.cat([F1, F2], dim=1)))
        return x1*Ws+x2*(1-Ws)
    

class PMFI(nn.Module):
    def __init__(self, channel=64):
        super(PMFI, self).__init__()
        self.conv1 = BasicConv2d(channel, channel)
        self.conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(2*channel, channel)
        self.conv4 = BasicConv2d(3*channel, channel)
        self.conv5 = BasicConv2d(4*channel, channel)
        self.conv6 = nn.Conv2d(channel, 1, kernel_size=1)
        self.cat = Cat()
        self.FF = FFusion()
    def forward(self, x1, x2, x3, x4):
        h1,w1 = x1.size()[-2:]
        h2,w2 = x2.size()[-2:]
        h3,w3 = x3.size()[-2:]
        
        F1 = self.conv2(x1)
        F2 = self.conv2(x2)
        F3 = self.conv2(x3)
        F4 = self.conv2(x4)
        
        F4_1 = F4
        F4_o = self.FF(F4_1,F4)
        F4_u = F.interpolate(F4_o,size=(h3,w3), mode='bicubic', align_corners=False)
        
        F3_c = self.conv3(torch.cat([F4_u, F3], dim=1))
        F3_1 = self.conv2(F3)
        F3_o = self.FF(F3_c,F3_1)
        F3_u = F.interpolate(F3_o,size=(h2,w2), mode='bicubic', align_corners=False)
        
        out_refine = self.cat(x1, x2)
        out_refine_2 = F.interpolate(out_refine,size=(h2,w2), mode='bicubic', align_corners=False)
        
        F4_o_2 = F.interpolate(F4_o,size=(h2,w2), mode='bicubic', align_corners=False)
        F2_c = self.conv4(torch.cat([F2, out_refine_2+F3_u+out_refine_2*F3_u, F4_o_2], dim=1))
        F2_1 = self.conv2(F2)
        F2_o = self.FF(F2_c,F2_1)
        F2_u = F.interpolate(F2_o,size=(h1,w1), mode='bicubic', align_corners=False)
        
        F3_o_2 = F.interpolate(F3_o,size=(h1,w1), mode='bicubic', align_corners=False)
        F1_c = self.conv5(torch.cat([F1, out_refine+F2_u+out_refine*F2_u, F2_u, F3_o_2], dim=1))
        F1_1 = self.conv2(F1)
        F1_o = self.FF(F1_c,F1_1)
        
        edge = self.conv6(F1_o)
        return F1_o, edge
    
    def initialize(self):
        weight_init(self)
        
class Encoder(nn.Module):
    def __init__(self, channel=64):
        super(Encoder, self).__init__()
        self.a1 = Fre_D(channel, channel)
        self.a2 = Fre_D(channel, channel)
        self.a3 = Fre_D(channel, channel)
        self.a4 = Fre_D(channel, channel)
        
        self.LLFE = LLFE()
        self.HLFE = HLFE()
        self.pmfi = PMFI()
        self.ma = nn.MaxPool2d(2)
        
        self.conv2 = BasicConv2d(channel, 3, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(channel, 3, kernel_size=3, padding=1)
        
    def forward(self, x1, x2, x3, x4):  ##x1->x2->x3
        res1h,res1w = x1.size()[-2:]
        res2h,res2w = x2.size()[-2:]
        
        yL1, yH1 = self.a1(x1)
        yL2, yH2 = self.a2(x2)
        yL3, yH3 = self.a3(x3)
        yL4, yH4 = self.a4(x4)
        
        F1 = self.LLFE(yL1, yH1)
        F1 = F.interpolate(F1,size=(res1h,res1w), mode='bicubic', align_corners=False)
        
        yH1_2 = self.ma(yH1)
        F2 = self.LLFE(yL2, yH1_2)
        F2 = F.interpolate(F2,size=(res2h,res2w), mode='bicubic', align_corners=False)
        
        yH2_2 = self.ma(yH2)
        F3 = self.HLFE(yL3, yH2_2, x3)
        
        yH3_2 = self.ma(yH3)
        F4 = self.HLFE(yL4, yH3_2, x4)
        
        ma, edge = self.pmfi(F1, F2, F3, F4)
        
        F3_2 = self.conv2(F3)
        F4_2 = self.conv3(F4)
        
        return F1, F2, F3, F4, ma, edge, F3_2, F4_2
    
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        h = self.relu(self.conv2(h))
        return h
        
class ADFD(nn.Module): # Progress Feature Interaction
    def __init__(self, channel=64):
        super(ADFD, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel + 16, channel, 3, padding=1), nn.ReLU(True),
        )
        self.conv6 = BasicConv2d(2, 1, kernel_size=3, padding=1)
        self.ed = nn.Conv2d(channel, 1, 3, padding=1)
        self.conv7 = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x, y):
        y1 = 1-torch.sigmoid(y)
        y2 = y
        x1 = x+self.conv1(x*y2)
        x1_1 = self.conv2(x1)
        
        xs_1 = torch.chunk(x1_1, 16, dim=1)
        x_f_1 = x1_1+self.conv4(torch.cat([xs_1[0], y1, xs_1[1], y1, xs_1[2], y1, xs_1[3], y1, xs_1[4], y1, xs_1[5], y1, xs_1[6], y1, xs_1[7], y1, 
                                          xs_1[8], y1, xs_1[9], y1, xs_1[10], y1, xs_1[11], y1, xs_1[12], y1, xs_1[13], y1, xs_1[14], y1, xs_1[15], y1], 1))
        edge_1 = y1+self.ed(x_f_1)
        
        x1_2 = self.conv3(x1)
        xs_2 = torch.chunk(x1_2, 16, dim=1)
        x_f_2 = x1_2+self.conv4(torch.cat([xs_2[0], y1, xs_2[1], y1, xs_2[2], y1, xs_2[3], y1, xs_2[4], y1, xs_2[5], y1, xs_2[6], y1, xs_2[7], y1,
                                          xs_2[8], y1, xs_2[9], y1, xs_2[10], y1, xs_2[11], y1, xs_2[12], y1, xs_2[13], y1, xs_2[14], y1, xs_2[15], y1], 1))
        edge_2 = y1+self.ed(x_f_2)
        
        Res_1 = torch.chunk(x_f_1, 16, dim=1)
        x_re_1 = x_f_1+self.conv4(torch.cat([Res_1[0], edge_2, Res_1[1], edge_2, Res_1[2], edge_2, Res_1[3], edge_2, Res_1[4], edge_2, Res_1[5], edge_2, Res_1[6], edge_2, 
                                            Res_1[7], edge_2, Res_1[8], edge_2, Res_1[9], edge_2, Res_1[10], edge_2, Res_1[11], edge_2, Res_1[12], edge_2, Res_1[13], edge_2,
                                            Res_1[14],edge_2, Res_1[15],edge_2], dim=1))
        edge_2_1 = edge_2+self.ed(x_re_1)
        
        Res_2 = torch.chunk(x_f_2, 16, dim=1)
        x_re_2 = x_f_2+self.conv4(torch.cat([Res_2[0], edge_1, Res_2[1], edge_1, Res_2[2], edge_1, Res_2[3], edge_1, Res_2[4], edge_1, Res_2[5], edge_1, Res_2[6], edge_1, 
                                             Res_2[7], edge_1, Res_2[8], edge_1, Res_2[9], edge_1, Res_2[10], edge_1, Res_2[11], edge_1, Res_2[12], edge_1, Res_2[13], edge_1,
                                             Res_2[14],edge_1, Res_2[15],edge_1], dim=1))
        edge_2_2 = edge_1+self.ed(x_re_2)
        
        out = self.conv7(edge_2_1+edge_2_2)+y1
        return out
    
    def initialize(self):
        weight_init(self)

class LoG(nn.Module):
    def __init__(self, channel=64):
        super(LoG, self).__init__()
        self.channel = channel
        self.kernel = [[0,1,1,2,2,2,1,1,0],
                        [1,2,4,5,5,5,4,2,1],
                        [1,4,5,3,0,3,5,4,1],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [2,5,0,-24,-40,-24,0,5,2],
                        [2,5,3,-12,-24,-12,3,5,2],
                        [1,4,5,3,0,3,4,4,1],
                        [1,2,4,5,5,5,4,2,1],
                        [0,1,1,2,2,2,1,1,0]]
        self.conv1 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        
    def forward(self, x):
        window = torch.FloatTensor(self.kernel).expand(self.channel, 1, 9, 9).contiguous().cuda()
        output = F.conv2d(x, window, padding = 4, groups = self.channel)
        output = self.conv1(output)
        return output

class EdgeDetection(nn.Module):
    def __init__(self, channel=64):
        super(EdgeDetection, self).__init__()
        self.log = LoG(channel)
        self.conv1 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(channel, 1, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(2*channel, channel, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x2 = F.interpolate(x2, size=(h,w), mode='bicubic', align_corners=False)
        x1, x2 = self.log(x1), self.log(x2)
        Fc = self.conv4(torch.cat([x1,x2], dim=1))
        sc = self.sigmoid(Fc)
        ca = sc*x2+x1
        ca2 = ca.transpose(-1,-2)
        ca2 = self.conv2(ca2)
        ca2 = ca2.transpose(-1,-2)
        out = ca2+ca
        return self.conv3(out)
    
    def initialize(self):
        weight_init(self)

class SSFusion(nn.Module):
    def __init__(self, channel=64):
        super(SSFusion, self).__init__()
        self.tsbd = TSBD(channel)
        self.conv1 = BasicConv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(1, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, F1, Edge):
        F1 = self.conv1(F1)
        Edge = self.conv2(Edge)
        out = F1*Edge+F1
        re = self.tsbd(out)+out
        return re
    
class Network(nn.Module):
    def __init__(self, channel=64, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.bkbone = res2net50_v1b_26w_4s(pretrained=True)
        self.conv1_1 = RFB_modified(256, channel)
        self.conv1_2 = RFB_modified(512, channel)
        self.conv1_3 = RFB_modified(1024, channel)
        self.conv1_4 = RFB_modified(2048, channel)
        self.encoder = Encoder(channel)
        self.conv2 = BasicConv2d(channel, 1, kernel_size=1)
        self.RS4 = ADFD(channel)
        self.RS3 = ADFD(channel)
        self.RS2 = ADFD(channel)
        self.RS1 = ADFD(channel)
        self.ssfu1 = SSFusion(channel)
        self.ssfu2 = SSFusion(channel)
        self.edgeD = EdgeDetection(channel)
        self.conv3 = nn.Conv2d(1,1, kernel_size=1)
    def forward(self, x):
        b, c, h, w = x.size()
        h2, w2 = x.size()[-2:]
        
        x = self.bkbone.conv1(x)
        x = self.bkbone.bn1(x)
        x = self.bkbone.relu(x)
        x = self.bkbone.maxpool(x)
        
        x1 = self.bkbone.layer1(x)
        x2 = self.bkbone.layer2(x1)
        x3 = self.bkbone.layer3(x2)
        x4 = self.bkbone.layer4(x3)
        
        x1 = self.conv1_1(x1) #64
        x2 = self.conv1_2(x2) #64
        x3 = self.conv1_3(x3) #64
        x4 = self.conv1_4(x4) #64
        
        edge1_1 = self.edgeD(x1,x2) ##b 1 h w
        pre_edge1_1 = F.interpolate(edge1_1, size=(h2,w2), mode='bicubic', align_corners=False) 
        
        F1, F2, F3, F4, ma, edge, F3_2, F4_2 = self.encoder(x1, x2, x3, x4)
        F1 = self.ssfu1(F1, edge1_1)
        
        b, c, h1, w1 = F4.size()
        edge2 = F.interpolate(edge, size=(h1,w1), mode='bicubic', align_corners=False)
        
        re4 = self.RS4(F4,edge2)
        re4_2 = self.conv3(re4 + edge2) ####
        
        b, c, h, w = F3.size()
        pre_3 = F.interpolate(re4_2, size=(h,w), mode='bicubic', align_corners=False)
        
        re3 = self.RS3(F3, pre_3)
        re3_2 = self.conv3(re3 + pre_3) ####
        
        b, c, h, w = F2.size()
        edge2_2 = F.interpolate(edge1_1, size=(h,w), mode='bicubic', align_corners=False)
        F2 = self.ssfu1(F2, edge2_2)
        pre_2 = F.interpolate(re3_2, size=(h,w), mode='bicubic', align_corners=False)
        
        re2 = self.RS2(F2, pre_2)
        re2_2 = self.conv3(re2 + pre_2) ####
        
        b, c, h, w = F1.size()
        pre_1 = F.interpolate(re2_2, size=(h,w), mode='bicubic', align_corners=False)
        
        re1 = self.RS1(F1, pre_1)
        re1_2 = self.conv3(re1 + pre_1)  ####
        
        p4 = F.interpolate(re4_2,size=(h2,w2), mode='bicubic', align_corners=False)
        p3 = F.interpolate(re3_2,size=(h2,w2), mode='bicubic', align_corners=False)
        p2 = F.interpolate(re2_2,size=(h2,w2), mode='bicubic', align_corners=False)
        p1 = F.interpolate(re1_2,size=(h2,w2), mode='bicubic', align_corners=False)
        peg = F.interpolate(edge,size=(h2,w2), mode='bicubic', align_corners=False)

        return p1, p2, p3, p4, peg, F3_2, F4_2, pre_edge1_1
    
if __name__ == '__main__':
    x = torch.rand([4,3,256,256]).to('cuda')
    model = Network().to('cuda')
    print(model(x)[0].shape)