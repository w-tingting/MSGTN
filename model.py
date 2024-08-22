import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GATv2Conv
from layers import Conv3x3BNReLU, SSN

np.set_printoptions(threshold=np.inf)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myGNN(torch.nn.Module):
    def __init__(self, num_inputfeatures: int, num_outfeatures: int):
        super().__init__()
        self.BN = nn.BatchNorm1d(num_inputfeatures)
        self.conv1 = GATv2Conv(num_inputfeatures, 64, heads=4)
        self.conv2 = GATv2Conv(256, num_outfeatures, heads=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.BN(x)
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        return x
class myGIN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGIN, self).__init__()
        self.args = args
        self.conv1 = GINConv(
            Sequential(
                Linear(in_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                BN(hidden_dim),
            ), train_eps=False)
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_dim, hidden_dim),
                ReLU(),
                Linear(hidden_dim, out_dim),
                ReLU(),
                BN(out_dim),
            ), train_eps=False)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        print(x.shape)
        print(edge_index.shape)
        x = self.conv1(x, edge_index)
        print(x.shape)
        x = self.conv2(x, edge_index)
        node_embeddings = x

        return node_embeddings

class myGAT(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGAT, self).__init__()
        self.args = args
        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, dropout=0.5)

        self.conv2 = GATConv(hidden_dim * 4, out_dim, heads=1,
                             concat=False, dropout=0.5)
        self.relu = torch.nn.LeakyReLU(0.2)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)

        node_embeddings = x
        return node_embeddings


class myGCN(torch.nn.Module):
    def __init__(self, args, in_dim, out_dim, hidden_dim):
        super(myGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        # self.fc1 = torch.nn.Linear(out_dim, 9)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        # x = self.fc1(x)

        node_embeddings = x

        return node_embeddings


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''

    def __init__(self, in_ch, out_ch, num_depth_conv_layer, kernel_size=5):
        super(SSConv, self).__init__()
        self.num_depth_conv_layer = num_depth_conv_layer
        self.depth_conv = nn.Sequential()
        for i in range(self.num_depth_conv_layer):
            self.depth_conv.add_module('depth_conv_' + str(i), nn.Conv2d(in_channels=out_ch, out_channels=out_ch,
                                                                         kernel_size=kernel_size, stride=1,
                                                                         padding=kernel_size // 2, groups=out_ch))

        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )
        self.Act = nn.LeakyReLU(inplace=True)
        self.BN = nn.BatchNorm2d(in_ch)

    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act(out)
        for i in range(self.num_depth_conv_layer):
            out = self.depth_conv[i](out)
            out = self.Act(out)
        return out


class MultiGraphTrans(nn.Module):

    def __init__(self, args, spec_band, num_classes, layer_num, num_spixels=[128, 256, 512], init_weights=True,
                 in_channel=256):
        super(MultiGraphTrans, self).__init__()

        self.args = args
        self.layer_num = layer_num
        self.superpixel_count = num_spixels  # 256
        self.num_classes = num_classes  # 16
        self.num_nodes_features = in_channel  # 128

        self.backbone = nn.Sequential(
            Conv3x3BNReLU(spec_band, 64),  # (204,64)
            # nn.MaxPool2d(kernel_size=1, stride=1),
            Conv3x3BNReLU(64, 128),
            Conv3x3BNReLU(128, 64),  # 128,128
        )

        if init_weights:
            self._initialize_weights()
        self.num_spixels = num_spixels
        self.ssn = SSN(args, self.num_spixels)

        input_num = 64
        hidden_num = 16
        output_num = 32
        self.GAT_layers = nn.Sequential()
        for i in range(self.layer_num):
            self.GAT_layers.add_module('my_GNN_l' + str(i), myGCN(self.args, in_dim=input_num, out_dim=output_num,
                                                                  hidden_dim=hidden_num).to(device))

            # input_num = input_num + output_num
            # input_num = input_num
            # output_num = input_num
        self.fc1 = torch.nn.Linear(input_num + output_num, self.num_classes)
    def forward(self, x):
        _, b, h, w = x.size()
        # feature space
        x = self.backbone(x)  # 1,256,610,340

        # superspixel segments
        # Q:(H*W, superspixel_num); S:(real_num_spixel, num_spixels); A:(real_num_spixel, real_num_spixel)
        Q_list, edge_index_list, superpixels_flatten_list = self.ssn(x)
        # print("Q_list[0]", Q_list[0].shape)
        # print("Q_list[1]", Q_list[1].shape)
        # print("Q_list[2]", Q_list[2].shape)
        #
        # print("superpixels_flatten_list[0]", superpixels_flatten_list[0].shape)
        # print("superpixels_flatten_list[1]", superpixels_flatten_list[1].shape)
        # print("superpixels_flatten_list[2]", superpixels_flatten_list[2].shape)
        Gout = []
        Q = []
        for i in range(self.layer_num):
            H_i = self.GAT_layers[i](superpixels_flatten_list[i], edge_index_list[i])
            # print("H_i:", H_i.shape)
            tmp = torch.cat([superpixels_flatten_list[i], H_i], dim=-1)
            Gout.append(tmp)
            if i == 0:
                Q.append(Q_list[i])
            else:
                b, num_spixel, f = Q_list[i].shape
                Q_sclae = (F.interpolate(Q_list[i].reshape(b, num_spixel, h // (i * 2), w // (i * 2)), size=(h, w),
                                         mode='bilinear', align_corners=False)).reshape(b, num_spixel, -1)
                Q.append(Q_sclae)

        # print("Gout[0]:", Gout[0].shape)
        # print("Gout[1]:", Gout[1].shape)
        # print("Gout[2]:", Gout[2].shape)
        # print("Gout[3]:", Gout[3].shape)

        for i in range(len(Gout)):
            if i == 0:
                H_i = Gout[i]
            else:
                H_i = torch.cat((H_i, Gout[i]), dim=1)
        # H_i = torch.cat((Gout[0], Gout[1], Gout[2]), dim=1)
        # print("H_i:", H_i.shape)
        for i in range(len(Q)):
            if i == 0:
                Q_tmp = Q[i]
            else:
                Q_tmp = torch.cat((Q_tmp, Q[i]), dim=1)
        Q = Q_tmp
        # print("Q:", Q.shape)
        # Q = torch.cat((Q[0], Q[1], Q[2]), dim=1)
        Q_T = (Q.squeeze() / (torch.sum(Q.squeeze(), 0, keepdim=True))).transpose(0, 1)

        H_i = self.fc1(H_i)
        return H_i, Q_T

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
