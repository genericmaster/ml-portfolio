import torch as pt

class InputEmbedding(pt.nn.Module):
    def __init__(self,vocab_size,batch_size,embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedded_batch = None
        self.dict_embedding = pt.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.input_embedding =None
        
    def embedding(self,batch_matrix):
        embeddings= self.dict_embedding
        self.embedded_batch = embeddings(batch_matrix)    
        return self.embedded_batch
    def positional_encoding(self,batch_embedding):
        postional_matrix = pt.zeros_like(batch_embedding)
        intermediate_calc = batch_embedding.shape[1]
        position_sin = pt.arange(intermediate_calc).unsqueeze(1)
        positions_cos = pt.arange(intermediate_calc).unsqueeze(1)
        intermediate_calc2= self.embedding_dim//2
        sin_embedding_dim = pt.arange(intermediate_calc2)
        cos_embedding_dim = pt.arange(intermediate_calc2)
        
        postional_matrix[:,:,0:intermediate_calc2]= pt.sin(position_sin/1000**((2*sin_embedding_dim)/self.embedding_dim))
        postional_matrix[:,:,intermediate_calc2:]= pt.cos(positions_cos/1000**((2*cos_embedding_dim)/self.embedding_dim))
        
        self.input_embedding = batch_embedding+postional_matrix
        
        return self.input_embedding

