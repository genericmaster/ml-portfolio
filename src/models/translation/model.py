import torch as pt
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
from .encoder import Encoder
from .decoder import DECODER
from .embeddings import InputEmbedding


class Transformer(pt.nn.Module):
    def __init__(self, embedding_dim, heads, blocks, vocab_size_enc, vocab_size_dec, batch_size):
        super().__init__()
        self.encoder_embedding = InputEmbedding(vocab_size=vocab_size_enc, batch_size=batch_size, embedding_dim=embedding_dim)
        self.decoder_embedding = InputEmbedding(vocab_size=vocab_size_dec, batch_size=batch_size, embedding_dim=embedding_dim)
        self.Encoderblocks = Encoder(embedding_dim=embedding_dim, heads=heads, blocks=blocks)
        self.Decoderblocks = DECODER(embedding_dim=embedding_dim, heads=heads, blocks=blocks)
        self.linear = pt.nn.Linear(in_features=embedding_dim, out_features=vocab_size_dec)
        
    def forward(self, source_batch, target_batch,src_mask,trg_mask):
        # encoder
        encoder_embed = self.encoder_embedding.embedding(source_batch)
        encoder_matrix = self.encoder_embedding.positional_encoding(encoder_embed)
        encoder_context_matrix = self.Encoderblocks.encoder_forward(encoder_matrix,src_mask)

        # decoder
        decoder_embed = self.decoder_embedding.embedding(target_batch)
        decoder_matrix = self.decoder_embedding.positional_encoding(decoder_embed)
        decoder_context_matrix = self.Decoderblocks(decoder_matrix, encoder_context_matrix,src_mask,trg_mask)

        # linear
        logits = self.linear(decoder_context_matrix)
        return logits
