import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, p_drop=0.4):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation, padding=dilation)
        self.conv_res = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        out = self.dilated_conv(x)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        res = self.conv_res(out)
        skip = self.conv_skip(out)
        
        if res.size(2) != x.size(2):
            res = res[:, :, :x.size(2)]
        if skip.size(2) != x.size(2):
            skip = skip[:, :, :x.size(2)]
        
        return res + x, skip

class WaveNet(nn.Module):
    def __init__(self, in_channels, res_channels, num_classes, dilation_cycles=10, layers_per_cycle=10, p_drop=0.4):
        super(WaveNet, self).__init__()
        self.residual_blocks = nn.ModuleList()
        for cycle in range(dilation_cycles):
            for layer in range(layers_per_cycle):
                dilation = 2 ** layer
                self.residual_blocks.append(ResidualBlock(in_channels, res_channels, dilation, p_drop))
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.dropout = nn.Dropout(p_drop)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        min_len = min([skip.size(2) for skip in skip_connections])
        skip_connections = [skip[:, :, :min_len] for skip in skip_connections]
        out = sum(skip_connections)
        out = self.relu(out)
        out = self.final_conv(out)
        out = self.batch_norm(out)
        out = torch.mean(out, dim=2)
        out = self.dropout(out)
        out = self.fc(out)
        return out
