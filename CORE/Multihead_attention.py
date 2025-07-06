import torch
import torch.functional as F
import torch.nn as nn



class MHSelfAttention(nn.Module):
    def __init__(self, num_heads:int, d_model:int, mask=False, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.mask = mask
        assert d_model % num_heads == 0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        #defining the weight matrices
        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=bias).to(self.device)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=bias).to(self.device)
        self.W_V = nn.Linear(self.d_model, self.d_model,bias=bias).to(self.device)
        self.W_O = nn.Linear(self.d_model, self.d_model,bias=bias).to(self.device)
        
        
        
    def forward(self, x):
        b_size, seq_length, d_model = x.shape
        x.to(self.device)
        assert d_model == self.d_model
        # we first verify that the input has the same embedding dimension that the model expects
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        # we reshape our matrices in order to be able to apply attention
        d_head = d_model // self.num_heads
        Q = Q.reshape(b_size, seq_length, self.num_heads, d_head)
        Q = Q.transpose(1,2).reshape(b_size * self.num_heads, seq_length, d_head)
        K = K.reshape(b_size, seq_length, self.num_heads, d_head)
        K = K.transpose(1,2).reshape(b_size * self.num_heads, seq_length, d_head)
        V = V.reshape(b_size, seq_length, self.num_heads, d_head)
        V = V.transpose(1,2).reshape(b_size * self.num_heads, seq_length, d_head)
        # Calculating attention score
        att_scores = torch.bmm(Q, K.transpose(1,2))
        # Apply mask if we have it
        if self.mask:
            mask = torch.tril(torch.ones(seq_length, seq_length)).to(torch.bool).unsqueeze(0)
            att_scores.masked_fill(mask,float('-inf'))
        # Calculate the attention weights
        att_weights = torch.softmax(att_scores / torch.sqrt(torch.tensor(d_head, dtype=torch.float32)), dim=-1)
        # Calculating the final result
        out = torch.bmm(att_weights, V).reshape(b_size, self.num_heads, seq_length, d_head)
        out = out.transpose(1,2).reshape(b_size, seq_length, self.d_model)
        out = self.W_O(out)
        return out
    
        
        
        
        
if __name__ == '__main__':
    d_model = 10
    num_heads = 2
    x = torch.rand(1,4,d_model).to('cuda')
    model = MHSelfAttention(num_heads, d_model)
    print(x)
    y = model(x)
    print(y)
        
        

                  
              
        