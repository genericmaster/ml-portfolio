import torch as pt
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')

class InputEmbedding(pt.nn.Module):
    def __init__(self,vocab_size,batch_size,embedding_dim):
        super().__init__()
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.batch_list = []
        self.mask_list=[]
        self.embedded_batch = None
        self.dict_embedding = pt.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.input_embedding =None
    def padding(self,input):
        self.batch_list =[]
        self.mask_list=[]
        for batch in range(0,len(input),self.batch_size):
            chunk = input[batch:batch+self.batch_size]
            padded = pt.nn.utils.rnn.pad_sequence(chunk, batch_first=True, padding_value=0)
            mask = (padded == 0)
            self.batch_list.append(padded)
            self.mask_list.append(mask)
        return self.batch_list,self.mask_list
            
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

         
         
        
        
        