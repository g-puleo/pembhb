import torch
import torch.nn as nn
from lightning import LightningModule
from torch import nn
# class GWTransformer(LightningModule):
#     def __init__(self, n_chunks=100, embed_dim=512, num_heads=8, num_layers=4):
#         super().__init__()
#         self.n_chunks = n_chunks
#         self.chunk_size = 10000 // n_chunks
#         self.input_dim = 3 * self.chunk_size  # 3 channels × chunk length

#         # Input projection
#         self.embedding = nn.Linear(self.input_dim, embed_dim)

#         # Positional encoding
#         self.positional_encoding = nn.Parameter(torch.randn(n_chunks, embed_dim))

#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)



#     def forward(self, data, parameters):
#         """
#         x: Tensor of shape (batch_size, 3, 10000) containing the frequency domain signal in the 3 TDI channels
#         extra_features: Tensor of shape (batch_size, 11)
#         """
#         B, C, T = data.shape
#         assert C == 3 and T == self.n_chunks * self.chunk_size, "Invalid input shape"

#         # Split into chunks and reshape
#         data = data.view(B, C, self.n_chunks, self.chunk_size)
#         data = data.permute(0, 2, 1, 3).reshape(B, self.n_chunks, -1)  # (B, n_chunks, 3 * chunk_size)

#         # Embed and add positional encodings
#         x = self.embedding(data) + self.positional_encoding  # (B, n_chunks, embed_dim)

#         # Transformer encoding
#         x = self.transformer(x)  # (B, n_chunks, embed_dim)
#         x = x.mean(dim=1)  # global average pooling over sequence: (B, embed_dim)

#         # Process extra input features
#         extra = self.extra_fc(parameters)  # (B, embed_dim)

#         # Concatenate and classify
#         combined = torch.cat([x, extra], dim=-1)  # (B, embed_dim * 2)
#         out = self.classifier(combined)  # (B, 1)

#         return out

class MarginalClassifierHead(nn.Module):

    def __init__(self, in_features: int , marginals: list[list], hidden_size: int = 32):
        super().__init__()
        self.marginals = marginals
        self.classifiers = nn.ModuleList([
            nn.Sequential(
            nn.Linear(in_features + len(marginal), hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
            )
            for marginal in marginals
        ])

    def forward(self, features, parameters):
        outputs = []
        for i, marginal in enumerate(self.marginals):
            indices = marginal
            input_data = torch.cat([features, parameters[:, indices]], dim=-1)
            outputs.append(self.classifiers[i](input_data))
        return torch.cat(outputs, dim=-1)

class InferenceNetwork(LightningModule):
    """ 
    Basic FC network for TMNRE of MBHB data. 
    I wrote this function to test the swyft framework, improvements on the architecture are future work. 
    """
    def __init__(self, num_features=10000, num_channels  = 6,  hlayersizes=(500,20), lr=1e-3):  
        super().__init__()

        self.normalise = nn.BatchNorm1d(num_features=num_channels, eps=1e-22)
        self.conv1d = nn.Conv1d(num_channels, 1, kernel_size=3, padding=1) 
        self.fc_blocks = nn.Sequential()
        input_size = num_features
        for i, output_size in enumerate(hlayersizes):
            self.fc_blocks.add_module(f"fc_{i}", nn.Linear(input_size, output_size))
            self.fc_blocks.add_module(f"relu_{i}", nn.Sigmoid())
            self.fc_blocks.add_module(f"dropout_{i}", nn.Dropout(p=0.5))
            input_size = output_size
        
        self.logratios = MarginalClassifierHead(input_size, marginals=[[0,1]], hidden_size=20)
        self.loss = nn.BCEWithLogitsLoss(reduce='sum')
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x, parameters):
        """
        Forward pass of the network.
        
        Args:
            data: Tensor of shape (batch_size, num_channels, num_features) containing the frequency domain signal in the TDI channels
            parameters: Tensor of shape (batch_size, 11) containing the parameters to be used in the classifier
        """
        
        data = self.normalise(x)
        #print("data min max std mean", torch.min(data), torch.max(data), torch.std(data), torch.mean(data))
        data = self.conv1d(data)  
        #print("data after conv1d min max std mean", torch.min(data), torch.max(data), torch.std(data), torch.mean(data))
        data = data.squeeze(1)  # (batch_size, num_features - 2)
        features = self.fc_blocks(data)  # (batch_size, hidden_size)
        #print("features min max std mean", torch.min(features), torch.max(features), torch.std(features), torch.mean(features))
        output = self.logratios(features, parameters)  # (batch_size, num_marginals)
        #print("output min max std mean", torch.min(output), torch.max(output), torch.std(output), torch.mean(output))
        return output


    def training_step(self, batch, batch_idx): 
        data = batch['data_fd']
        parameters = batch['source_parameters']
        scrambled_params = torch.roll(parameters, shifts=1, dims=0)

        output_joint = self(data, parameters)
        output_scrambled = self(data, scrambled_params)
        loss_1 = self.loss(output_joint, torch.ones_like(output_joint))
        loss_2 = self.loss(output_scrambled, torch.zeros_like(output_scrambled))
        loss = loss_1 + loss_2
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        data = batch['data_fd']
        parameters = batch['source_parameters']
        scrambled_params = torch.roll(parameters, shifts=1, dims=0)

        output_joint = self(data, parameters)
        output_scrambled = self(data, scrambled_params)

        # Calculate losses
        loss_1 = self.loss(output_joint, torch.ones_like(output_joint))
        loss_2 = self.loss(output_scrambled, torch.zeros_like(output_scrambled))
        loss = loss_1 + loss_2

        # Calculate accuracy
        joint_preds = (torch.sigmoid(output_joint) > 0.5).float()
        scrambled_preds = (torch.sigmoid(output_scrambled) > 0.5).float()
        joint_accuracy = (joint_preds == 1).float().mean()
        scrambled_accuracy = (scrambled_preds == 0).float().mean()
        accuracy = (joint_accuracy + scrambled_accuracy) / 2

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
