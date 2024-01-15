import torch
import torch.nn as nn
from .encoding import PositionalEncoding

class PatchTST(nn.Module):
    def __init__(self, n_token, input_dim, model_dim, num_heads, num_layers, output_dim, dim_feedforward=256):
        super(PatchTST, self).__init__()
        self.patch_embedding = nn.Linear(input_dim, model_dim)    # Input Embedding
        self._pos = torch.nn.Parameter(torch.randn(1,1,model_dim))  # Positional Embedding
        #self._pos = PositionalEncoding(input_dim,model_dim,"cpu")
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, 
                                                    dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        self.output_layer = nn.Linear(model_dim * n_token, output_dim)

    def forward(self, x):
        # x shape: (batch_size, n_token, token_size)
        x = self.patch_embedding(x)   # (batch_size, n_token, model_dim)
        x = x + self._pos
        x = self.transformer_encoder(x)   # (batch_size, n_token, model_dim)
        x = x.view(x.size(0), -1)       # (batch_size, n_token * model_dim)
        output = self.output_layer(x)   # (batch_size, out_dim =4 patch_size == 4)
        return output
    
class PatchSRT(nn.Module):
    def __init__(self, n_token, input_dim, model_dim, num_heads, num_layers, output_dim, dim_feedforward=256):
        super(PatchSRT, self).__init__()
        self.patch_embedding = nn.Linear(input_dim, model_dim)    # Input Embedding
        self._pos = torch.nn.Parameter(torch.randn(1,1,model_dim))  # Positional Embedding
        #self._pos = PositionalEncoding(input_dim,model_dim,"cpu")
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, 
                                                    dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # feedforwarding layer
        self.output_layer = nn.Linear(model_dim, input_dim)
        
        # residual learning block
        self.activation = nn.LeakyReLU()
        self.res_layer_1 = nn.Linear(input_dim, model_dim*2)
        self.res_layer_2 = nn.Linear(model_dim*2, model_dim*2)
        self.res_layer_3 = nn.Linear(model_dim*2, input_dim)
        # # fully connected layer block
        self.res_layer_4 = nn.Linear(input_dim, input_dim)
        self.fc_layer = nn.Linear(input_dim*n_token, output_dim)

    def forward(self, x):
        # x shape: (batch_size, n_token, token_size)
        x_ = self.patch_embedding(x)   # (batch_size, n_token, model_dim)
        x_ = x_ + self._pos
        x_ = self.transformer_encoder(x_)   # (batch_size, n_token, model_dim)   
        x_ = self.output_layer(x_)   # (batch_size, out_dim =4 patch_size == 4)
        x = self.res_layer_1(x-x_)
        x = self.activation(x)
        x = self.res_layer_2(x)
        x = self.activation(x)
        x = self.res_layer_3(x)
        x = self.activation(x)
        #x = self.res_layer_4(x)
        #x = self.activation(x)
        # fully connected layer block
        x = self.res_layer_4(x_+x) 
        x = x.view(x.size(0), -1) # (batch_size, n_token * model_dim)
        x = self.fc_layer(x)
        return x