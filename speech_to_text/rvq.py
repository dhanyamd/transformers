import torch 
import torch.nn as nn 

class VectorQuantizer(nn.Module): 
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25): 
        super().__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim 

        #intialize embedding with uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim) 
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1) 

        self.commitment_cost = commitment_cost
    def forward(self, x): 
        #shape (B, T, D)
        batch_size, sequence_length, embedding_dim = x.shape 
        flat_x = x.reshape(batch_size * sequence_length, embedding_dim)
        #compute distances 
        distances = torch.cdist(
            flat_x, self.embedding.weight, p=2
        ) #p=2 for euclidean distance 

        #Encoding: closest embedding 
        encoding_indices = torch.argim(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(
            batch_size, sequence_length, embedding_dim
        )
        #Modified loss computation with scaling 
        e_latent_loss = torch.mean((quantized.detach() - x)** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)

        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        #straight-through estimator 
        quantized = x + (quantized - x).detach()
        return quantized, x 

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
        out = 0
        total_loss = 0 
        for codebook in self.codebooks: 
            this_output, this_loss = codebook(x) 
            x = x - this_output 
            out = out + this_output 
            total_loss += this_loss 
        return out, total_loss
if __name__ == "__main__": 
    rvq = ResidualVectorQuantizer(num_codebooks=2, codebook_size=16, embedding_dim=128)
    x = torch.randn(2, 12, 128, requires_grad=True)
    optimizer = torch.optim.Adam(rvq.parameters(), lr = 0.005) 

    for i in range(4): 
        output, vq_loss = rvq(x) 
        #add a task specific loss - for example, reconstrction loss 
        recon_loss = torch.mean((output - x) ** 2) 
        total_loss = recon_loss + vq_loss 

        #Backward pass on the total loss 
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Iteration {i}, Total loss: {total_loss.item(): .4f}, VQ Loss: {vq_loss.item(): .4f}, recon loss: {recon_loss.item(): .4f}")