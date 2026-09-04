import torch as pt
from .decoder import Decoder
from .embeddings import InputEmbedding

class CodingModel(pt.nn.Module):
    def __init__(self,vocab_size,batch_size, embedding_dim, heads, blocks):
        super().__init__()
        self.input_embedding = InputEmbedding(vocab_size=vocab_size,batch_size=batch_size, embedding_dim=embedding_dim)
        self.decoder_blocks = Decoder(heads=heads, blocks=blocks, embedding_dim=embedding_dim)
        self.linear = pt.nn.Linear(in_features=embedding_dim, out_features=vocab_size)

    def forward(self, x):
        embeddings = self.input_embedding.embedding(x)
        positional_embed = self.input_embedding.positional_encoding(embeddings)
        decoder_context_matrix = self.decoder_blocks.forward(positional_embed)
        logits = self.linear(decoder_context_matrix)
        return logits
    
    