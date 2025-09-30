import torch
import torch.nn as nn
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from typing import Iterable
from pembhb.utils import _ORDERED_PRIOR_KEYS
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
WEEK_SI = 7 * 24 * 3600  # seconds in a week
class MarginalClassifierHead(nn.Module):

    def __init__(self, n_data_features: int , marginals: list[list], hlayersizes: Iterable[int]):
        """Classifier head for multiple marginals. 
        Performs a binary classification for each marginal in the list of marginals. Used for TMNRE. 

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
            indices = marginal
            input_data = torch.cat([features, parameters[:, indices]], dim=-1)
            outputs.append(self.classifiers[i](input_data))

        return torch.cat(outputs, dim=-1)

class InferenceNetwork(LightningModule):
    """ 
    Basic FC network for TMNRE of MBHB data. 
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
        self.output_names = []
        for d_idx, domain in enumerate(self.marginals):
            for i, marginal in enumerate(self.marginals[domain]):
                # create a nice string for the marginal 
                name_output = ""
                for idx in marginal: 
                    name_output += str(_ORDERED_PRIOR_KEYS[idx]) + "_"
                name_output = name_output[:-1]  # remove trailing underscore
                self.output_names.append(name_output)

    def forward(self, d_f, d_t, parameters):
        """
        Forward pass of the network.
        
        Args:
            data: Tensor of shape (batch_size, num_channels, num_features) containing the frequency domain signal in the TDI channels
            parameters: Tensor of shape (batch_size, 11) containing the parameters to be used in the classifier
        """
        output = self.model(d_f, d_t, parameters)  # (batch_size, num_marginals)
        return output

    def calc_logits_losses(self, data_f, data_t, parameters):

        all_data_f = torch.cat((data_f, data_f), dim=0)
        all_data_t = torch.cat((data_t, data_t), dim=0)

        scrambled_params = torch.roll(parameters, shifts=1, dims=0)
        all_params = torch.cat((parameters, scrambled_params), dim=0)
        all_logits = self(all_data_f, all_data_t, all_params) 
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
        data_f = batch['data_fd']
        data_t = batch['data_td']
        parameters = batch['source_parameters']

        all_logits, loss_params, loss = self.calc_logits_losses(data_f, data_t, parameters)
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        
        self.log('train_loss', loss , on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        overall_idx = 0
        for d_idx, domain in enumerate(self.marginals.keys()):
            for i, marginal in enumerate(self.marginals[domain]):
                name = self.output_names[overall_idx]
                self.log(f'train_accuracy_{name}', accuracy_params[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log(f'train_loss_{name}', loss_params[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
                overall_idx += 1
        return loss

    def validation_step(self, batch, batch_idx):
        all_logits, loss_params, loss= self.calc_logits_losses(batch['data_fd'], batch["data_td"], batch['source_parameters'])
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        overall_idx = 0
        for d_idx, domain in enumerate(self.marginals):
            for i, marginal in enumerate(self.marginals[domain]):
                name = self.output_names[overall_idx]
                self.log(f'val_accuracy_{name}', accuracy_params[overall_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log(f'val_loss_{name}', loss_params[overall_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                overall_idx += 1
        return loss
    
    def test_step(self, batch, batch_idx):
        all_logits, loss_params, loss = self.calc_logits_losses(batch['data_fd'], batch['source_parameters'])
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        # print accuracies on test set
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        for d_idx, domain in enumerate(self.marginals):
            for i, marginal in enumerate(self.marginals[domain]):
                self.log(f'test_accuracy_{domain}_{i}', accuracy_params[i], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience, min_lr=1e-6)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 10**(epoch//15) if epoch < 30 else 100)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer,
        #     base_lr=0.0001,
        #     max_lr=0.001,
        #     step_size_up=15,
        # )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class SimpleModel(torch.nn.Module):
    def __init__(self, num_features: int, num_channels: int,  hlayersizes: tuple,  marginals: dict[list[list]], marginal_hidden_size: int, lr: float):  
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
        
        # Validate keys of marginals
        allowed_keys = {"f", "t", "ft"}
        invalid_keys = set(self.marginals.keys()) - allowed_keys
        if invalid_keys:
            raise ValueError(f"Invalid keys in marginals: {invalid_keys}. Allowed keys are {allowed_keys}.")
        
        self.bounds_trained = conf["prior"]

        n_channels =  len(conf["waveform_params"]["channels"])
        n_timesteps = int(conf["waveform_params"]["duration"]*WEEK_SI/conf["waveform_params"]["dt"])
        n_freqs = n_timesteps // 2 
        self.normalisation_f = nn.BatchNorm1d(num_features=n_channels*2)
        #self.normalisation_t = nn.BatchNorm1d(num_features=n_channels)
        self.unet_f = Unet(
            n_in_channels=n_channels*2,
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(2, 2, 2, 2),
        )
        self.unet_t = Unet(
            n_in_channels=n_channels,
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(8,8,8,8),
        )

        self.flatten = nn.Flatten(1)
        self.linear_t = LinearCompression(n_timesteps)
        self.linear_f = LinearCompression(n_freqs)
        self.logratios_model_dict = nn.ModuleDict()
        for key in self.marginals.keys():
            self.logratios_model_dict[key] = MarginalClassifierHead(
                n_data_features=16*len(key),# key can be any of "f", "t", "ft", resulting in double input size if ft. 
                marginals=self.marginals[key], 
                hlayersizes=(100,100,100,100,50,25,10)
            ).to(conf["device"])





    def forward(self, d_f, d_t, parameters):
        #print(f"d_f shape: {d_f.shape}")
        d_f_norm = self.normalisation_f(d_f)
        #d_t_norm = self.normalisation_t(d_t)
        #print(f"d_f_norm shape: {d_f_norm.shape}")
        d_f_processed = self.unet_f(d_f_norm)
        d_t_processed = self.unet_t(d_t)
        #print(f"d_f_processed shape: {d_f_processed.shape}")
        flattened_f = self.flatten(d_f_processed)
        flattened_t = self.flatten(d_t_processed)

        #print(f"flattened shape: {flattened.shape}")
        features_f = self.linear_f(flattened_f)
        features_t = self.linear_t(flattened_t)
        features_ft = torch.cat((features_f, features_t), dim=-1)
        
        features_dict = {
            "ft": features_ft,
            "f": features_f,
            "t": features_t
        }
        # print(f"features_t device: {features_t.device}")
        # print(f"parameters device: {parameters.device}")
        # print(f"features_f device: {features_f.device}")
        # print(f"features_ft device: {features_ft.device}")
        # #print(f"features_f shape: {features_f.shape}")

        logratios_1d_list = []
        for key in self.logratios_model_dict.keys():
            #breakpoint()
            logratios_1d_list.append(self.logratios_model_dict[key](features_dict[key], parameters))
        logratios_1d = torch.cat(logratios_1d_list, dim=-1)
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
