import torch.nn as nn 
import torch 
import torch.nn.functional as F # Import F for log_softmax
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
            max_seq_length: int = 2000 # Max sequence length for positional encoding in Transformer
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
            hidden_dim=embedding_dim, # Assuming hidden_dim in DownsamplingNetwork is embedding_dim
            strides=strides,
            initial_mean_pooling_kernel_size=intial_mean_pooling_kernel_size # Pass this to DownsamplingNetwork
        )
        self.pre_rvq_transformer = Transformer(
            embedding_dim,
            num_layers=num_transformer_layers,
            max_seq_length=max_seq_length # Ensure this max_seq_length is sufficient for downsampled audio
        )
        # CRITICAL FIX: Add LayerNorm before RVQ for stability
        # This normalizes the features before they enter the VQ-VAE, preventing exploding values.
        self.norm_before_rvq = nn.LayerNorm(embedding_dim) 
        self.rvq = ResidualVectorQuantizer(num_codebooks, codebook_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size) 
    
    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len) -> unsqueeze(1) -> (batch_size, 1, seq_len)
        x = x.unsqueeze(1) 
        
        # DownsamplingNetwork output: (batch_size, embedding_dim, downsampled_seq_len)
        # Note: DownsamplingNetwork already transposes at the end to (B, T, D)
        x = self.downsampling_network(x)
        
        # Transformer expects (batch_size, sequence_length, embedding_dim)
        x = self.pre_rvq_transformer(x) 
        
        # Apply LayerNorm before RVQ for numerical stability
        x = self.norm_before_rvq(x) # CRITICAL FIX: Apply LayerNorm here
        
        # RVQ output: (batch_size, downsampled_seq_len, embedding_dim), vq_loss
        x, vq_loss = self.rvq(x) 
        
        # Output layer: (batch_size, downsampled_seq_len, vocab_size)
        x = self.output_layer(x)
        
        # Apply log_softmax across the vocabulary dimension (dim=2)
        # This is crucial for CTC Loss, which expects log-probabilities over the vocabulary.
        x = F.log_softmax(x, dim=2) # Corrected dim=1 to dim=2 in previous turn
        
        return x, vq_loss 
    
    def save(self, path: str):
        print("Saving model to", path) 
        torch.save({"model" : self.state_dict(), "options": self.options}, path) 
    
    @staticmethod 
    def load(path: str):
        print("Loading model from", path)
        loaded_data = torch.load(path)
        model = TranscribeModel(**loaded_data["options"])
        model.load_state_dict(loaded_data["model"]) 
        return model 

if __name__ == "__main__": 
    # Example usage for testing TranscribeModel
    model = TranscribeModel(
          num_codebooks=3,
          codebook_size=64,
          embedding_dim=64,
          vocab_size=30,
          strides=[6,8,4,2],
          intial_mean_pooling_kernel_size=4,
          max_seq_length=2000, # This needs to be greater than the max output length of DownsamplingNetwork
          num_transformer_layers=2
    )
    # Example input audio: (batch_size, raw_audio_length)
    x = torch.randn(4, 237680) 
    out, loss = model(x)
    print("Model Output Shape:", out.shape)
    print("VQ Loss from model:", loss.item()) # Print VQ loss for inspection
