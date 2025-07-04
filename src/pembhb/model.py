import torch
import torch.nn as nn
import swyft
from swyft.networks import Network
class GWTransformer(swyft.SwyftModule):
    def __init__(self, n_chunks=100, embed_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        self.n_chunks = n_chunks
        self.chunk_size = 10000 // n_chunks
        self.input_dim = 3 * self.chunk_size  # 3 channels × chunk length

        # Input projection
        self.embedding = nn.Linear(self.input_dim, embed_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(n_chunks, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



    def forward(self, data, parameters):
        """
        x: Tensor of shape (batch_size, 3, 10000) containing the frequency domain signal in the 3 TDI channels
        extra_features: Tensor of shape (batch_size, 11)
        """
        B, C, T = data.shape
        assert C == 3 and T == self.n_chunks * self.chunk_size, "Invalid input shape"

        # Split into chunks and reshape
        data = data.view(B, C, self.n_chunks, self.chunk_size)
        data = data.permute(0, 2, 1, 3).reshape(B, self.n_chunks, -1)  # (B, n_chunks, 3 * chunk_size)

        # Embed and add positional encodings
        x = self.embedding(data) + self.positional_encoding  # (B, n_chunks, embed_dim)

        # Transformer encoding
        x = self.transformer(x)  # (B, n_chunks, embed_dim)
        x = x.mean(dim=1)  # global average pooling over sequence: (B, embed_dim)

        # Process extra input features
        extra = self.extra_fc(parameters)  # (B, embed_dim)

        # Concatenate and classify
        combined = torch.cat([x, extra], dim=-1)  # (B, embed_dim * 2)
        out = self.classifier(combined)  # (B, 1)

        return out

class InferenceNetwork(swyft.SwyftModule):
    """ 
    Basic FC network for TMNRE of MBHB data. 
    I wrote this function to test the swyft framework, improvements on the architecture are future work. 
    """
    def __init__(self, num_features=10000, hlayersizes=(500,20)):  
        super().__init__()
        self.fc_blocks = nn.Sequential()
        input_size = num_features
        for i, output_size in enumerate(hlayersizes):
            self.fc_blocks.add_module(f"fc_{i}", nn.Linear(input_size, output_size))
            self.fc_blocks.add_module(f"relu_{i}", nn.ReLU())
            self.fc_blocks.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            input_size = output_size
        
        self.logratios1 = swyft.LogRatioEstimator_1dim(num_features = 20, num_params = 11, varnames = 'z_tot')
        self.logratios2 = swyft.LogRatioEstimator_Ndim(num_features = 20, marginals=((0,1),(7,8)))
    def forward(self, A, B):
        processed_x = self.fc_blocks(A["data_fd"])
        logratios1 = self.logratios1(processed_x, B["z_tot"])
        logratios2 = self.logratios2(processed_x, B["z_tot"])
        return logratios1, logratios2


 
    
