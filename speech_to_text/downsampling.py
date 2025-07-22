import torch.nn.functional as F 
from torch import nn 
import torch
class ResidualDownSampleBlock(nn.Module): 
    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()
        self.conv1 = nn.Conv1d(
        in_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(out_channels) 
        self.conv2 = nn.Conv1d(
        out_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride 
        )
        self.relu = nn.ReLU() 
    def forward(self, x): 
        #x: (batch_size, in_channels, seq_len)
        output = self.conv1(x) 
        output = self.bn1(output) 
        output = self.relu(output) + x 
        output = self.conv2(output)
        return output
class DownsamplingNetwork(nn.Module): 
    def __init__(self, embedding_dim=128, hidden_dim=64, in_channels=1, initial_mean_pooling_kernel_size=2, strides=[6,6,8,4,2]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.mean_pooling = nn.AvgPool1d(kernel_size=initial_mean_pooling_kernel_size)
        for i in range(len(strides)): 
            self.layers.append(
                ResidualDownSampleBlock(
                    hidden_dim if i > 0 else in_channels,
                    hidden_dim,
                    strides[i],
                    kernel_size=8
                )
            )
        self.final_conv = nn.Conv1d(
            hidden_dim, embedding_dim, kernel_size=4, padding="same"
        )
    def forward(self, x): 
        print("x shape", x.shape)
        x = self.mean_pooling(x)
        print("mean pooling shape", x.shape) 
        for i, layer in enumerate(self.layers):
            x = layer(x) 
            print(f"layer {i} shape")
        x = self.final_conv(x)
        print("final conv shape", x.shape) 
        x = x.transpose(1,2)
        print("transpose shape", x.shape)
        return x 

if __name__ == "__main__": 
    batch_size = 2 
    input_embedding_dim = 1 
    seq_len = 237680 
    hidden_dim = 16
    output_embedding_dim = 32 
    strides = [2,4,8]
    initial_mean_pooling_kernel_size = 2
    downsampling_network = DownsamplingNetwork(
        embedding_dim=output_embedding_dim,
        hidden_dim=hidden_dim,
        in_channels=input_embedding_dim,
        initial_mean_pooling_kernel_size=initial_mean_pooling_kernel_size,
        strides=strides
    )
    x = torch.randn(batch_size, 1, seq_len)
    print(downsampling_network(x).shape)