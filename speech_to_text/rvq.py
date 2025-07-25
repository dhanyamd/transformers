import torch 
import torch.nn as nn 
import torch.nn.functional as F # Added F import for potential future use or clarity

class VectorQuantizer(nn.Module): 
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25): 
        super().__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim 

        # Initialize embedding with uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim) 
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1) 

        self.commitment_cost = commitment_cost
        
    def forward(self, x): 
        # x shape: (B, T, D) where B=batch_size, T=sequence_length, D=embedding_dim
        batch_size, sequence_length, embedding_dim = x.shape 
        
        # Reshape x to (B*T, D) for cdist calculation
        flat_x = x.reshape(batch_size * sequence_length, embedding_dim)
        
        # Compute Euclidean distances from each flattened input vector to all codebook embeddings
        distances = torch.cdist(
            flat_x, self.embedding.weight, p=2 # p=2 for Euclidean distance
        ) 

        # Encoding: find the index of the closest embedding for each input vector
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Quantize: retrieve the actual embedding vectors using the indices
        # .view() reshapes back to original (B, T, D)
        quantized = self.embedding(encoding_indices).view(
            batch_size, sequence_length, embedding_dim
        )
        
        # Loss computation for VQ-VAE:
        # 1. Commitment Loss (e_latent_loss): Measures how far the encoder output (x) is from the chosen codebook vector (quantized).
        #    We detach 'quantized' here so gradients flow only through 'x' (encoder).
        e_latent_loss = torch.mean((quantized.detach() - x)** 2)
        
        # 2. Codebook Loss (q_latent_loss): Measures how far the codebook vector (quantized) is from the encoder output (x).
        #    We detach 'x' here so gradients flow only through 'quantized' (codebook embeddings).
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)

        # Total VQ loss: A weighted sum of codebook loss and commitment loss.
        # Both q_latent_loss and e_latent_loss are squared differences, so they are always positive.
        # Therefore, 'loss' should always be positive.
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator:
        # During forward pass, 'quantized' is used.
        # During backward pass, gradients are passed directly through 'x' (encoder output)
        # as if no quantization happened, by adding (quantized - x) and detaching it.
        quantized = x + (quantized - x).detach()
        
        # CRITICAL FIX: Return the calculated 'loss' instead of the input 'x'
        return quantized, loss # MODIFIED: Returned 'loss' instead of 'x'

class ResidualVectorQuantizer(nn.Module): 
    def __init__(self, num_codebooks, codebook_size, embedding_dim): 
        super().__init__() 
        self.codebooks = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim)
                for _ in range(num_codebooks) 
            ]
        )
        
    def forward(self, x): 
        out = 0 # Accumulates the quantized outputs from each codebook
        total_loss = 0 # Accumulates the loss from each codebook
        
        for codebook in self.codebooks: 
            # this_output: quantized output from the current codebook
            # this_loss: VQ loss from the current codebook (should be positive)
            this_output, this_loss = codebook(x) 
            
            # Update x for the next codebook: subtract the quantized output
            # This is the "residual" part of Residual VQ
            x = x - this_output 
            
            # Accumulate the quantized outputs
            out = out + this_output 
            
            # Accumulate the losses
            total_loss += this_loss 
            
        return out, total_loss

if __name__ == "__main__": 
    # Example usage for testing ResidualVectorQuantizer
    rvq = ResidualVectorQuantizer(num_codebooks=2, codebook_size=16, embedding_dim=128)
    x = torch.randn(2, 12, 128, requires_grad=True) # Example input tensor
    optimizer = torch.optim.Adam(rvq.parameters(), lr = 0.005) 

    print("--- Testing ResidualVectorQuantizer ---")
    for i in range(4): 
        output, vq_loss = rvq(x) # Forward pass
        
        # Add a task-specific loss (e.g., reconstruction loss) for a complete VQ-VAE setup
        # In a full VQ-VAE, 'output' would be fed to a decoder to reconstruct the original 'x'.
        # Here, we just use the direct difference for demonstration.
        recon_loss = torch.mean((output - x) ** 2) 
        
        # Total loss for this test: reconstruction loss + VQ loss
        total_loss = recon_loss + vq_loss 

        # Backward pass and optimizer step
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print current loss values
        print(f"Iteration {i}, Total loss: {total_loss.item(): .4f}, VQ Loss: {vq_loss.item(): .4f}, recon loss: {recon_loss.item(): .4f}")

    print("--- Test Complete ---")
