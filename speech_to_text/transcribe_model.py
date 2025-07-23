import torch.nn as nn 
import torch 
from rvq import ResidualVectorQuantizer, VectorQuantizer 
from self_attention import Transformer
from downsampling import DownsamplingNetwork 

class TranscribeModel(nn.Module): 
    def __init__(
            self, 
            num_codebooks: int,
            codebook_size: int,
            embedding_dim : int,
            vocab_size: int,
            strides: list[int],
            intial_mean_pooling_kernel_size: int,
            num_transformer_layers: int,
            max_seq_length: int = 2000
    ): 
        super().__init__()
        self.options = {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "embedding_dim": embedding_dim,
            "vocab_size": vocab_size,
            "strides": strides,
            "num_transformer_layers": num_transformer_layers,
            "intial_mean_pooling_kernel_size": intial_mean_pooling_kernel_size,
            "max_seq_length": max_seq_length
        }
        self.downsampling_network = DownsamplingNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            strides=strides 
        )
        self.pre_rvq_transformer = Transformer(
            embedding_dim,
            num_layers=num_transformer_layers,
            max_seq_length=max_seq_length
        )
        self.rvq = ResidualVectorQuantizer(num_codebooks, codebook_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size) 
    
    def forward(self, x: torch.Tensor):
        loss = torch.tensor(0.0)
        x = x.unsqueeze(1)
        x = self.downsampling_network(x)
        x = self.pre_rvq_transformer(x)
        x, loss = self.rvq(x) 
        x = self.output_layer(x)
        x = torch.log_softmax(x, dim=1)
        return x, loss 
    def save(self, path: str):
        print("Saving model to", path) 
        torch.save({"model" : self.state_dict(), "options": self.options}, path) 
    @staticmethod 
    def load(path: str):
        print("Loading model from", path)
        model = TranscribeModel(**torch.load(path)["options"])
        model.load_state_dict(torch.load(path)["model"]) 
        return model 

if __name__ == "__main__": 
    model = TranscribeModel(
          num_codebooks=3,
          codebook_size=64,
          embedding_dim=64,
          vocab_size=30,
          strides=[6,8,4,2],
          intial_mean_pooling_kernel_size=4,
          max_seq_length=2000,
          num_transformer_layers=2
    )
    x = torch.randn(4, 237680)
    out, loss = model(x)
    print(out.shape)