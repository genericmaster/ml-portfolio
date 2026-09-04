import torch as pt
class MaskedMultiHeadAttention(pt.nn.Module):
    def __init__(self, heads, embedding_dim):
        super().__init__()
        self.heads = heads
        self.embedding = embedding_dim
        self.head_dim = self.embedding // self.heads
        self.query = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.key = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.value = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.projection = pt.nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

    def forward(self, input):
        sequence_length = input.shape[1]
        batch_size = input.shape[0]
        query = self.query(input)
        self.q = pt.reshape(query, shape=(batch_size, sequence_length, self.heads, self.head_dim)).permute(0, 2, 1, 3)
        key = self.key(input)
        self.k = pt.reshape(key, shape=(batch_size, sequence_length, self.heads, self.head_dim)).permute(0, 2, 1, 3)
        value = self.value(input)
        self.v = pt.reshape(value, shape=(batch_size, sequence_length, self.heads, self.head_dim)).permute(0, 2, 1, 3)

        attention_score = (pt.matmul(self.q, self.k.permute(0, 1, 3, 2))) / (self.head_dim ** 0.5)
        masked_matrix = pt.triu(pt.ones(sequence_length, sequence_length, dtype=pt.bool), diagonal=1)
        attention_score = attention_score.masked_fill(masked_matrix, float('-inf'))
        attention_weights = pt.softmax(attention_score, dim=-1)
        context_vector = pt.matmul(attention_weights, self.v).permute(0, 2, 1, 3).contiguous().reshape(batch_size, sequence_length, self.embedding)
        output = self.projection(context_vector)
        return output



class LayerNorm(pt.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.gamma = pt.nn.Parameter(pt.ones(embedding_dim))
        self.beta = pt.nn.Parameter(pt.zeros(embedding_dim))

    def forward(self, input):
        mean = input.mean(dim=-1, keepdim=True)
        std = input.std(dim=-1, keepdim=True)
        normalization = (input - mean) / (std + 1e-08)
        normalized_embedding = (normalization * self.gamma) + self.beta
        return normalized_embedding



class FeedFoward(pt.nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.layer1 = pt.nn.Linear(in_features=embedding_dim, out_features=4 * embedding_dim)
        self.layer2 = pt.nn.Linear(in_features=4 * embedding_dim, out_features=embedding_dim)

    def forward(self, input):
        first_layer = self.layer1(input)
        activation = pt.relu(first_layer)
        second_layer = self.layer2(activation)
        return second_layer
    
    

#main purpose is to be an insatiation of a layer within the network
class DecoderBlock(pt.nn.Module):
    def __init__(self,heads,embedding_dim):
        super().__init__()
        self.self_attention = MaskedMultiHeadAttention(heads=heads,embedding_dim=embedding_dim)
        self.layer_norm_1 = LayerNorm(embedding_dim=embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim=embedding_dim)
        self.feed_foward = FeedFoward(embedding_dim=embedding_dim)
        
    def forward(self,x):
         # Pre-LN attention
        norm1 = self.layer_norm_1(x)
        attention_output = self.self_attention(norm1)
        residual = attention_output + x
        # Pre-LN feedforward
        norm2 = self.layer_norm_2(residual)
        mlp_output = self.feed_foward(norm2)
        output = residual + mlp_output
        return output
        
class Decoder(pt.nn.Module):
    def __init__(self,heads,blocks,embedding_dim):
        super().__init__()
        self.decoder_blocks_list = pt.nn.ModuleList([ DecoderBlock(embedding_dim=embedding_dim, heads=heads) for _ in range(blocks)])
        
    def forward(self,x):
        current = x
        for  decoder in self.decoder_blocks_list:
            current = decoder.forward(current)
        return current
        
        