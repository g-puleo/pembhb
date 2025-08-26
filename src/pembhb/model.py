import torch
import torch.nn as nn
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from typing import Iterable
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

    def __init__(self, n_data_features: int , marginals: list[list], hlayersizes: Iterable[int]):
        """_summary_

        :param n_data_features: the number of features in the data summary
        :type n_data_features: int
        :param marginals: list of marginals that you want 
        :type marginals: list[list]
        :param hidden_size: sizes of the hidden layers in the classifier
        :type hidden_size: iterable[int]
        """
        super().__init__()
        self.marginals = marginals
        self.classifiers = nn.ModuleList()
        for marginal in marginals:
            classifier = nn.Sequential()
            for i, output_size in enumerate(hlayersizes):
                if i == 0:
                    input_size = n_data_features + len(marginal)
                    output_size = hlayersizes[i]
                else:
                    input_size = hlayersizes[i-1]
                    output_size = hlayersizes[i]
        
                classifier.add_module(f"fc_{i}", nn.Linear(input_size, output_size))
                classifier.add_module(f"relu_{i}", nn.ReLU())

            classifier.add_module("output", nn.Linear(output_size, 1)) 
            self.classifiers.append(classifier)
        


    def forward(self, features, parameters):
        outputs = []
        for i, marginal in enumerate(self.marginals):
            # breakpoint()
            indices = marginal
            input_data = torch.cat([features, parameters[:, indices]], dim=-1)
            outputs.append(self.classifiers[i](input_data))

        return torch.cat(outputs, dim=-1)

class InferenceNetwork(LightningModule):
    """ 
    Basic FC network for TMNRE of MBHB data. 
    I wrote this function to test the swyft framework, improvements on the architecture are future work. 
    """
    def __init__(self,  conf: dict):  
        super().__init__()
        self.model = PeregrineModel(conf)
        self.marginals = self.model.marginals
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.lr = conf["training"]["learning_rate"]
        self.bounds_trained = self.model.bounds_trained
        self.scheduler_patience = conf["training"]["scheduler_patience"]
        self.scheduler_factor = conf["training"]["scheduler_factor"]
        self.save_hyperparameters(conf)
        
    def forward(self, x, parameters):
        """
        Forward pass of the network.
        
        Args:
            data: Tensor of shape (batch_size, num_channels, num_features) containing the frequency domain signal in the TDI channels
            parameters: Tensor of shape (batch_size, 11) containing the parameters to be used in the classifier
        """
        # breakpoint()
        output = self.model(x, parameters)  # (batch_size, num_marginals)
        return output

    def calc_logits_losses(self, data, parameters):

        all_data = torch.cat((data, data), dim=0)
        scrambled_params = torch.roll(parameters, shifts=1, dims=0)
        all_params = torch.cat((parameters, scrambled_params), dim=0)
        all_logits = self(all_data, all_params) 
        shape_logits_half = (all_logits.shape[0] // 2, all_logits.shape[1])
        labels = torch.cat((torch.ones(shape_logits_half, device="cuda"), torch.zeros(shape_logits_half, device="cuda")), dim=0)
        all_loss= self.loss(all_logits, labels)
        loss_params = torch.mean(all_loss, dim=0)
        loss = torch.mean(loss_params)
        return all_logits, loss_params, loss
    
    def calc_accuracies(self, all_logits):
        # the first half of all_params contains parameters from the joint, associated with the first half of all_data
        # the second half of all_params contains parameters from the marginal, independent of the second half of all_data
        shape_logits_half = (all_logits.shape[0] // 2, all_logits.shape[1])
        joint_preds = (all_logits[:shape_logits_half[0]] > 0).float()
        scrambled_preds = (all_logits[shape_logits_half[0]:] > 0).float()
        joint_accurate = (joint_preds == 1).float()
        scrambled_accurate = (scrambled_preds == 0).float()
        joint_accuracy_params = torch.mean(joint_accurate, dim=0)
        scrambled_accuracy_params = torch.mean(scrambled_accurate, dim=0)
        accuracy_params = (joint_accuracy_params + scrambled_accuracy_params) / 2
        accuracy = torch.mean(accuracy_params)

        return accuracy_params, accuracy

    def training_step(self, batch, batch_idx): 
        data = batch['data_fd']
        parameters = batch['source_parameters']

        all_logits, loss_params, loss = self.calc_logits_losses(data, parameters)
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        
        self.log('train_loss', loss , on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        for i, marginal in enumerate(self.marginals):
            self.log(f'train_accuracy_{i}', accuracy_params[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'train_loss_{i}', loss_params[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        all_logits, loss_params, loss= self.calc_logits_losses(batch['data_fd'], batch['source_parameters'])
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for i, marginal in enumerate(self.marginals):
            self.log(f'val_accuracy_{i}', accuracy_params[i], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'val_loss_{i}', loss_params[i], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        all_logits, loss_params, loss = self.calc_logits_losses(batch['data_fd'], batch['source_parameters'])
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        # print accuracies on test set
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        for i, marginal in enumerate(self.marginals):
            self.log(f'test_accuracy_{i}', accuracy_params[i], on_step=False, on_epoch=True)
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 10**(epoch//15) if epoch < 45 else 1000)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler
            }
        }

class SimpleModel(torch.nn.Module):
    def __init__(self, num_features: int, num_channels: int,  hlayersizes: tuple,  marginals: list[list], marginal_hidden_size: int, lr: float):  
        super().__init__()

        # self.normalise = nn.BatchNorm1d(num_features=num_channels, eps=1e-22)
        # self.conv1d = nn.Conv1d(num_channels, 1, kernel_size=3, padding=1) 
        self.fc_blocks = nn.Sequential()
        input_size = num_features
        for i, output_size in enumerate(hlayersizes):
            self.fc_blocks.add_module(f"fc_{i}", nn.Linear(input_size, output_size))
            self.fc_blocks.add_module(f"relu_{i}", nn.ReLU())
            #self.fc_blocks.add_module(f"dropout_{i}", nn.Dropout(p=0.2))
            input_size = output_size
        
        self.logratios = MarginalClassifierHead(input_size, marginals=marginals, hidden_size=marginal_hidden_size)
        self.marginals = marginals 
        

    def forward(self, x, parameters):
        """
        Forward pass of the network.
        
        Args:
            data: Tensor of shape (batch_size, num_channels, num_features) containing the frequency domain signal in the TDI channels
            parameters: Tensor of shape (batch_size, 11) containing the parameters to be used in the classifier
        """
        # data = self.normalise(x)
        # #print("data min max std mean", torch.min(data), torch.max(data), torch.std(data), torch.mean(data))
        # data = self.conv1d(data)  
        # #print("data after conv1d min max std mean", torch.min(data), torch.max(data), torch.std(data), torch.mean(data))
        # data = data.squeeze(1)  # (batch_size, num_features - 2)
        features = self.fc_blocks(x)  # (batch_size, hidden_size)
        #print("features min max std mean", torch.min(features), torch.max(features), torch.std(features), torch.mean(features))
        output = self.logratios(features, parameters)  # (batch_size, num_marginals)
        
        #print("output min max std mean", torch.min(output), torch.max(output), torch.std(output), torch.mean(output))
        return output


class PeregrineModel(torch.nn.Module):
    def __init__(self, conf: dict):
        super().__init__()
        self.batch_size = conf["training"]["batch_size"]
        self.marginals = conf["tmnre"]["marginals"]
        self.bounds_trained = conf["prior"]
        # self.unet_t = Unet(
        #     n_in_channels=2 * len(conf["waveform_params"]["TDI"]),
        #     n_out_channels=1,
        #     sizes=(16, 32, 64, 128, 256),
        #     down_sampling=(8, 8, 8, 8),
        # )
        n_channels = 2 * len(conf["waveform_params"]["TDI"])
        n_freqs = conf["waveform_params"]["n_freqs"]
        self.normalisation = nn.BatchNorm1d(num_features=n_channels)
        self.unet_f = Unet(
            n_in_channels=n_channels,
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(2, 2, 2, 2),
        )

        self.flatten = nn.Flatten(1)
        #self.linear_t = LinearCompression()
        self.linear_f = LinearCompression(n_freqs)

        self.logratios_1d = MarginalClassifierHead(
            n_data_features=16,
            marginals=self.marginals, 
            hlayersizes=(32, 16, 8, 4)
        )



    def forward(self, d_f, parameters):
        #print(f"d_f shape: {d_f.shape}")
        d_f_norm = self.normalisation(d_f)
        #print(f"d_f_norm shape: {d_f_norm.shape}")
        d_f_processed = self.unet_f(d_f_norm)
        #print(f"d_f_processed shape: {d_f_processed.shape}")
        flattened = self.flatten(d_f_processed)
        #print(f"flattened shape: {flattened.shape}")
        features_f = self.linear_f(flattened)
        #print(f"features_f shape: {features_f.shape}")
        logratios_1d = self.logratios_1d(features_f, parameters)
        #print(f"logratios_1d shape: {logratios_1d.shape}")
        return logratios_1d

### THIS CHUNK OF CODE IS DIRECTLY COPIED FROM PEREGRINE
# 1D Unet implementation below
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        mid_channels=None,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(down_sampling), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_signal_length = x2.size()[2] - x1.size()[2]

        x1 = F.pad(
            x1, [diff_signal_length // 2, diff_signal_length - diff_signal_length // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_out_channels,
        sizes=(16, 32, 64, 128),
        down_sampling=(2, 2, 2, 2),
    ):
        super(Unet, self).__init__()
        self.inc = DoubleConv(n_in_channels, sizes[0])
        self.down1 = Down(sizes[0], sizes[1], down_sampling[0])
        self.down2 = Down(sizes[1], sizes[2], down_sampling[1])
        self.down3 = Down(sizes[2], sizes[3], down_sampling[2])
        self.down4 = Down(sizes[3], sizes[4], down_sampling[3])
        self.up1 = Up(sizes[4], sizes[3])
        self.up2 = Up(sizes[3], sizes[2])
        self.up3 = Up(sizes[2], sizes[1])
        self.up4 = Up(sizes[1], sizes[0])
        self.outc = OutConv(sizes[0], n_out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        f = self.outc(x)
        return f


class LinearCompression(nn.Module):
    def __init__(self, N_input):
        super(LinearCompression, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(N_input, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def forward(self, x):
        return self.sequential(x)
