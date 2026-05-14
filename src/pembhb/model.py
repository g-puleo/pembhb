import torch
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
# from line_profiler import profile
import torch.nn as nn
from lightning import LightningModule
from torch import nn
from torch.nn import functional as F
from typing import Iterable
from pembhb.data import MBHBDataset
from torch.utils.data import DataLoader

from pembhb.utils import _ORDERED_PRIOR_KEYS, mbhb_collate_fn
from pembhb import ROOT_DIR, get_torch_dtype
import numpy as np
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


def build_data_summary(conf):
    ''' 
    conf must be from config.yaml "data_summary" section
    '''
    ds_type = conf["type"]
    init_kwargs = conf[ds_type]
    cls = DATA_SUMMARY_REGISTRY[conf["type"]]
    return cls(**init_kwargs)

class GradnormHandler : 

    def __init__(self, parent_module: 'InferenceNetwork', config: dict ):
        """Initialises gradnorm handler

        :param parent_module: the parent lightning module (in lightningmodule __init__ call, it is 'self')
        :type parent_module: LightningModule
        :param config: must contain the following keys: 'alpha' , 'common_params', 
        :type config: dict
        """
        self.model = parent_module
        self.alpha=config["alpha"]
        self.shared_layer = getattr(self.model.data_summary, config["common_params"])
        self.shared_params_gradnorm = [p for p in self.shared_layer.parameters() if p.requires_grad]
        self.was_called_once = False
        self.model.automatic_optimization=False 
    
    def training_step(self, batch, batch_idx):
        """Performs a training step.

        :param batch: The input batch of data.
        :type batch: dict
        :param batch_idx: The index of the batch.
        :type batch_idx: int
        :return: _tuple: all_logits, task_losses, total_loss
        :rtype: tuple
        """
        
        data_f = batch['data_fd']
        data_t = batch['data_td']
        parameters = batch['source_parameters']

        # same base call, but handler decides weighting logic
        all_logits, task_losses, total_loss, gradnorm_loss = self.calc_logits_losses_gradnorm(data_f, data_t, parameters)
        #update weights w_i (actually we store their logits to ensure positivity)
        optimizer_nre, optimizer_gradnorm = self.model.optimizers()
        optimizer_gradnorm.zero_grad()
        self.model.manual_backward(gradnorm_loss, retain_graph=True)
        optimizer_gradnorm.step()
        #update model params
        optimizer_nre.zero_grad()
        self.model.manual_backward(total_loss)
        optimizer_nre.step()

        self.model.log_dict(
            {
            "L_grad": gradnorm_loss,
            "weighted_loss": total_loss
            })
        weights_normalised = torch.nn.functional.softmax(self.model.weights_loss_logits, dim=0)
        for i, w_i in enumerate(weights_normalised):
            self.model.log(f"weight_{i}", w_i.detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return  all_logits, task_losses, total_loss

    def compute_gradnorm_loss(self, task_losses_all):
        ########## GRADNORM LOGIC ###############
        weights = torch.nn.functional.softmax(self.model.weights_loss_logits, dim=0)
        task_losses = torch.mean(task_losses_all, dim=0) 
        weighted_task_losses = weights * task_losses
        total_loss = torch.mean(weighted_task_losses)
        #compute gradient norms G_W^{(i)}
        G_W = []
        for i, wL_i in enumerate(weighted_task_losses):
            grads = torch.autograd.grad(
                outputs=wL_i,
                inputs=self.shared_params_gradnorm,
                retain_graph=True, 
                create_graph=True
            )
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grads]), p=2)
            G_W.append(grad_norm)
            self.model.log(f"gradient_norm_{self.model.output_names[i]}", grad_norm, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        G_W = torch.stack(G_W)
        G_bar = G_W.mean()

        #compute r_i(t) and GradNorm loss
        loss_ratios = task_losses / self.initial_losses  # L_i(t) / L_i(0)
        inv_rate = loss_ratios / loss_ratios.mean()      # r_i(t)
        targets = G_bar * (inv_rate ** self.alpha)
        #breakpoint()
        gradnorm_loss = torch.abs(G_W - targets.detach()).sum()  # treat targets as constant
        self.model.log("gradient_norm", G_bar, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.model
        return total_loss, gradnorm_loss


    def calc_logits_losses_gradnorm(self, data_f, data_t, parameters):
        '''
        returns: 
            all_logits: tensor of shape (2*B, num_marginals)
            tails_losses: tensor of shape (num_marginals,), the non weighted losses
            weighted_tails_losses: tensor of shape (num_marginals,) the weighted losses, for sparing computation
            weights: tensor of shape (num_marginals,), the weights used for the weighted loss (after softmax)
            loss: scalar tensor, the full weighted loss'''
        all_logits, all_loss = self.model._calc_logits_base(data_f, data_t, parameters)
        task_losses = torch.mean(all_loss, dim=0)
        if not self.was_called_once:
            print("Setting initial losses for GradNorm")
            self.initial_losses = task_losses.detach()
        self.was_called_once = True
        total_loss, gradnorm_loss = self.compute_gradnorm_loss(task_losses)
        return all_logits, task_losses, total_loss, gradnorm_loss

class MarginalClassifierHead(nn.Module):

    def __init__(self, n_data_features: int | list[int], marginals: list[list], hlayersizes: Iterable[int]):
        """Classifier head for multiple marginals.
        Performs a binary classification for each marginal in the list of marginals. Used for TMNRE.

        :param n_data_features: the number of features in the data summary.
            If an int, the same value is used for all marginals (shared data summary).
            If a list[int], each marginal gets its own data feature size
            (e.g. for per-parameter encoders where 2D marginals receive
            concatenated bottlenecks).
        :type n_data_features: int | list[int]
        :param marginals: list of marginals that you want , in the reparametrised index space
        :type marginals: list[list]
        :param hidden_size: sizes of the hidden layers in the classifier
        :type hidden_size: iterable[int]
        """
        super().__init__()
        self.marginals_dict = marginals
        if isinstance(n_data_features, int):
            n_data_features_list = [n_data_features] * len(marginals)
        else:
            n_data_features_list = list(n_data_features)
        self.classifiers = nn.ModuleList()
        for marginal, n_df in zip(marginals, n_data_features_list):
            classifier = nn.Sequential()
            for i, output_size in enumerate(hlayersizes):
                if i == 0:
                    input_size = n_df + len(marginal)
                    output_size = hlayersizes[i]
                else:
                    input_size = hlayersizes[i-1]
                    output_size = hlayersizes[i]

                classifier.add_module(f"fc_{i}", nn.Linear(input_size, output_size))
                classifier.add_module(f"relu_{i}", nn.ReLU())

            classifier.add_module("output", nn.Linear(output_size, 1))
            self.classifiers.append(classifier)
        


    def forward(self, features, parameters):
        """Compute logratios for all marginals 

        Args:
            features : compressed version of the data
            parameters : parameters to be used , already reparametrised (expanded number of columns for periodic bc)

        Returns:
            torch.Tensor: logratios for this marginal
        """
        outputs = []
        for i, marginal in enumerate(self.marginals_dict):
            indices = marginal
            input_data = torch.cat([features, parameters[:, indices]], dim=-1)
            outputs.append(self.classifiers[i](input_data))
            
        return torch.cat(outputs, dim=-1)



def reparametrise_periodic_bc(parameters, position_indices: list):
    """Replace each column in position_indices with (sin, cos) pair.

    Args:
        parameters: Tensor of shape (B, N)
        position_indices: list of column indices (in the original N-dim space) to reparametrise

    Returns:
        Tensor of shape (B, N + len(position_indices))
    """
    parameters_out = parameters
    offset = 0
    for idx in position_indices:
        shifted_idx = idx + offset
        sin_col = torch.sin(parameters_out[:, shifted_idx:shifted_idx+1])
        cos_col = torch.cos(parameters_out[:, shifted_idx:shifted_idx+1])
        parameters_out = torch.cat(
            [parameters_out[:, :shifted_idx], sin_col, cos_col, parameters_out[:, shifted_idx+1:]],
            dim=-1,
        )
        offset += 1
    return parameters_out


def normalise_sincos_cols(params_expanded, periodic_bc_params, param_index_remapping, sincos_mean, sincos_std):
    """Normalize the sin/cos columns in the expanded parameter tensor.

    After reparametrise_periodic_bc() each periodic parameter index i has been
    replaced by two columns [sin, cos] tracked in param_index_remapping[i].
    This function standardizes those columns using pre-computed statistics.

    Args:
        params_expanded: Tensor of shape (B, N + len(periodic_bc_params))
        periodic_bc_params: list of original periodic parameter indices
        param_index_remapping: dict mapping original index → [sin_col, cos_col]
        sincos_mean: 1-D tensor of length 2*len(periodic_bc_params), ordered
            [sin_mean_0, cos_mean_0, sin_mean_1, cos_mean_1, ...]
        sincos_std: same shape as sincos_mean

    Returns:
        Tensor same shape as params_expanded with sin/cos columns normalized.
    """
    params_out = params_expanded.clone()
    for k, idx in enumerate(periodic_bc_params):
        sin_col, cos_col = param_index_remapping[idx]
        params_out[:, sin_col] = (params_out[:, sin_col] - sincos_mean[2 * k]) / sincos_std[2 * k]
        params_out[:, cos_col] = (params_out[:, cos_col] - sincos_mean[2 * k + 1]) / sincos_std[2 * k + 1]
    return params_out


class InferenceNetwork(LightningModule):
    """ 
    Basic FC network for TMNRE of MBHB data. 
    """
    def __init__(self,  train_conf: dict,  dataset_info: dict, normalisation: dict,  data_summarizer: nn.Module=None, periodic_bc_params: list=None):  
        super().__init__()
        if data_summarizer is not None:
            self.data_summary = data_summarizer
        else:
            self.data_summary = build_data_summary(train_conf["architecture"]["data_summary"])
        
        self.n_features_summary = self.data_summary.get_n_features() 
        self.marginals_dict = train_conf["marginals"]
        self.loss = nn.BCEWithLogitsLoss(reduction='none')
        self.lr = train_conf["learning_rate"]
        # Use the actual sampling prior (sampler_init_kwargs) as the
        # authoritative source.  Fall back to conf["prior"] for backward
        # compatibility with old YAML sidecars that may lack the key.
        _sik = dataset_info.get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            self.bounds_trained = _sik["prior_bounds"]
        else:
            self.bounds_trained = dataset_info["conf"]["prior"]
        self.scheduler_patience = train_conf["scheduler_patience"]
        self.scheduler_factor = train_conf["scheduler_factor"]
        self.output_names = []
        self.marginals_list = []
        for d_idx, domain in enumerate(self.marginals_dict):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                # create a nice string for the marginal 
                name_output = ""
                for idx in marginal: 
                    name_output += str(_ORDERED_PRIOR_KEYS[idx]) + "_"
                name_output = name_output[:-1]  # remove trailing underscore
                self.output_names.append(name_output)
                self.marginals_list.append(marginal)
        _td_norm = normalisation["td_normalisation"]
        self.td_normalisation = _td_norm.item() if _td_norm is not None else None
        self.param_mean = torch.tensor(normalisation["param_mean"], dtype=get_torch_dtype()).to(train_conf["device"])
        self.param_std = torch.tensor(normalisation["param_std"], dtype=get_torch_dtype()).to(train_conf["device"])
        print("Parameter mean:", self.param_mean.shape)
        print("Parameter std:", self.param_std.shape)
        self.save_hyperparameters(logger=True)    
        
        self.use_gradnorm = train_conf["architecture"]["gradnorm"]["enabled"]
        if self.use_gradnorm:
            self.gradnorm = GradnormHandler(self, train_conf["architecture"]["gradnorm"])
            self.weights_loss_logits = torch.nn.Parameter(torch.zeros(len(self.marginals_list)))
            self.lr_gradnorm = train_conf["architecture"]["gradnorm"]["learning_rate"]


        # Build index remapping for periodic BC reparametrisation.
        # Each periodic parameter column is replaced by (sin, cos), expanding the tensor by 1 per param.
        if periodic_bc_params is None:
            periodic_bc_params = []
        self.periodic_bc_params = periodic_bc_params
        # Normalization is a no-op for periodic params (they are passed raw to sin/cos).
        self.param_mean[periodic_bc_params] = 0
        self.param_std[periodic_bc_params] = 1

        # sin/cos normalization statistics (identity fallback for backward compat)
        _n_periodic = len(periodic_bc_params)
        _sc_mean = normalisation.get("sincos_mean", [0.0] * (2 * _n_periodic))
        _sc_std  = normalisation.get("sincos_std",  [1.0] * (2 * _n_periodic))
        self.register_buffer("sincos_mean", torch.tensor(_sc_mean, dtype=get_torch_dtype()))
        self.register_buffer("sincos_std",  torch.tensor(_sc_std,  dtype=get_torch_dtype()))

        self.param_index_remapping = {}
        offset = 0
        for idx in range(len(_ORDERED_PRIOR_KEYS)):
            # when a parameter is passed 
            if idx in self.periodic_bc_params:
                self.param_index_remapping[idx] = [idx + offset, idx + offset + 1]  # map to the two new columns
                offset += 1
            else: 
                self.param_index_remapping[idx] = idx + offset 

        # for each marginal list in the marginal dict, map it to the new indices:
        self.marginals_dict_remapped = {}
        
        for key in self.marginals_dict.keys():
            marginals_remapped = []
            # remember that self.marginals_dict[key] is a list of marginals
            for marginal in self.marginals_dict[key]:
                current_marginal_remapped =  [] 
                for idx in marginal:
                    if idx in self.periodic_bc_params:
                         # add the two new columns, extend is a function that appends multiple elements at once. 
                        current_marginal_remapped.extend(self.param_index_remapping[idx])
                    else:
                        current_marginal_remapped.append(self.param_index_remapping[idx])
                marginals_remapped.append(current_marginal_remapped)
            
            self.marginals_dict_remapped[key] = marginals_remapped

        self.logratios_model_dict = nn.ModuleDict()
        for key in self.marginals_dict_remapped.keys():
            self.logratios_model_dict[key] = MarginalClassifierHead(
                n_data_features=self.n_features_summary*len(key),
                marginals=self.marginals_dict_remapped[key], 
                hlayersizes=train_conf.get("classifier_hlayersizes", (64, 32, 16, 8))
            ).to(train_conf["device"])

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Backward compatibility: old checkpoints don't have sincos_mean/sincos_std
        # buffers. Fall back to the identity-normalization values built in __init__
        # (mean=0, std=1) so load_state_dict doesn't fail on missing keys.
        for buf_name in ("sincos_mean", "sincos_std"):
            key = prefix + buf_name
            if key not in state_dict and hasattr(self, buf_name):
                state_dict[key] = getattr(self, buf_name)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def transform_td(self, data_t):
        """Applies time-domain normalisation to the input data.

        :param data_t: Tensor of shape (batch_size, num_channels, num_timebins) containing the time domain signal in the TDI channels
        :type data_t: torch.Tensor
        :return: Normalised time-domain data.
        :rtype: torch.Tensor
        """
        return data_t / self.td_normalisation
    
    def transform_log(self, complex_data):
        """Applies log transformation to the complex input data, with separate channels for amplitude and phase.
        """
        amplitude = torch.abs(complex_data)
        phase = torch.angle(complex_data)
        log_amplitude = torch.log(amplitude+1e-33)
        # cat 
        transformed_data = torch.cat((log_amplitude, phase), dim=1)
        return transformed_data 
    
    def forward(self, d_f, d_t, parameters):
        """
        Forward pass of the network.
        
        Args:
            d_f: complex Tensor of shape (batch_size, num_channels, num_freqbins) containing the frequency domain signal in the TDI channels.
            d_t: real Tensor of shape (batch_size, num_channels, num_timebins) containing the time domain signal in the TDI channels
            parameters: Tensor of shape (batch_size, 11) containing the parameters to be used in the classifier
        """
        # preprocessing 
        # here, d_f is a complex array of raw data (wave + noise), and d_t is real array of raw data (wave + noise)
        d_f, d_t = self.data_summary(d_f, d_t)

        features_ft = None
        
        features_dict = {
            "ft": features_ft,
            "f": d_f,
            "t": d_t
        }
        #  normalise the non angular parameters: 
        # this line does not touch the parameters in periodic_bc_params because their mean is set to 0 and std to 1
        normalised_parameters = (parameters - self.param_mean) / self.param_std
        # take sin and cos of the params specified in periodic_bc_params
        reparametrised_withbc_params = reparametrise_periodic_bc(normalised_parameters, self.periodic_bc_params)
        if len(self.periodic_bc_params) > 0:
            reparametrised_withbc_params = normalise_sincos_cols(
                reparametrised_withbc_params, self.periodic_bc_params,
                self.param_index_remapping, self.sincos_mean, self.sincos_std,
            )
        logratios_1d_list = []
        for key in self.logratios_model_dict.keys():
            logratios_output = self.logratios_model_dict[key](features_dict[key], reparametrised_withbc_params)
            logratios_1d_list.append(logratios_output)
        logratios_1d = torch.cat(logratios_1d_list, dim=-1)
        return logratios_1d

    def _calc_logits_base(self, batch):
        data_f = batch['wave_fd']+batch["noise_fd"]
        if "wave_td" in batch:
            data_t = batch['wave_td']+batch["noise_td"]
            all_data_t = torch.cat((data_t, data_t), dim=0)
        else:
            all_data_t = None
        parameters = batch['source_parameters']
        all_data_f = torch.cat((data_f, data_f), dim=0)
        scrambled_params = torch.roll(parameters, shifts=1, dims=0)
        all_params = torch.cat((parameters, scrambled_params), dim=0)
        all_logits = self(all_data_f, all_data_t, all_params)
        shape_half = (all_logits.shape[0] // 2, all_logits.shape[1])
        labels = torch.cat(
            (torch.ones(shape_half, device=self.device, dtype=get_torch_dtype()), torch.zeros(shape_half, device=self.device, dtype=get_torch_dtype())),
            dim=0,
        )
        all_loss = self.loss(all_logits, labels)
        return all_logits, all_loss
    

    def calc_logits_losses(self, batch):
        all_logits, all_loss = self._calc_logits_base(batch)
        task_losses = torch.mean(all_loss, dim=0)
        loss = torch.mean(task_losses)
        return all_logits, task_losses, loss

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
        # these are complex representations
        if self.use_gradnorm:
            all_logits, task_losses, loss = self.gradnorm.training_step(batch, batch_idx)
        else:
            all_logits, task_losses, loss = self.calc_logits_losses(batch)

        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        self.log('train_loss', loss , on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        overall_idx = 0
        for d_idx, domain in enumerate(self.marginals_dict.keys()):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                name = self.output_names[overall_idx]
                self.log(f'train_accuracy_{name}', accuracy_params[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log(f'train_loss_{name}', task_losses[i], on_step=True, on_epoch=True, prog_bar=True, logger=True)
                overall_idx += 1

        if not self.use_gradnorm:
            return loss
    
    def validation_step(self, batch, batch_idx):
        # if self.use_gradnorm:
        #     all_logits, task_losses, weighted_losses_tails, weights,  loss= self.gradnorm.calc_logits_losses_gradnorm(batch['data_fd'], batch["data_td"], batch['source_parameters'])
        # else:
        all_logits, task_losses, loss= self.calc_logits_losses(batch)
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        overall_idx = 0
        for d_idx, domain in enumerate(self.marginals_dict):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                name = self.output_names[overall_idx]
                self.log(f'val_accuracy_{name}', accuracy_params[overall_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log(f'val_loss_{name}', task_losses[overall_idx], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                overall_idx += 1
        return loss
    
    def test_step(self, batch, batch_idx):
        all_logits, task_losses, loss = self.calc_logits_losses(batch['data_fd'], batch['source_parameters'])
        accuracy_params, accuracy = self.calc_accuracies(all_logits)
        # print accuracies on test set
        self.log('test_acc', accuracy, on_step=False, on_epoch=True)
        for d_idx, domain in enumerate(self.marginals_dict):
            for i, marginal in enumerate(self.marginals_dict[domain]):
                self.log(f'test_accuracy_{domain}_{i}', accuracy_params[i], on_step=False, on_epoch=True)

    def configure_optimizers(self):

        if self.use_gradnorm:
            trainable_params = [p for n, p in self.named_parameters() if n != "weights_loss_logits"]
            optimizer_nre = torch.optim.AdamW(trainable_params, lr=self.lr)
            optimizer_gradnorm = torch.optim.Adam([self.weights_loss_logits], lr=self.lr_gradnorm)
            return optimizer_nre, optimizer_gradnorm

        else: 
            optimizer_nre = torch.optim.AdamW(self.parameters(), lr=self.lr)
            return optimizer_nre
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.scheduler_factor, patience=self.scheduler_patience, min_lr=1e-6)
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 10**(epoch//15) if epoch < 30 else 100)
        #scheduler = torch.optim.lr_scheduler.CyclicLR(
        #     optimizer ,
        #     base_lr=0.0001,
        #     max_lr=0.001,
        #     step_size_up=15,
        # )
        

class PerMarginalInferenceNetwork(InferenceNetwork):
    """InferenceNetwork variant where each *parameter* has its own dedicated
    data encoder (ConvEncoder trained with parameter regression).

    Architecture
    ------------
    Each unique parameter p gets its own encoder:

        noisy_FD  →  ConvEncoder_p  →  bottleneck_p

    For a 1D marginal [p]:
        (bottleneck_p ‖ params_p)  →  classifier  →  logratio

    For a 2D marginal [p, q]:
        (bottleneck_p ‖ bottleneck_q ‖ params_p ‖ params_q)  →  classifier  →  logratio

    The per-parameter encoders are produced by :class:`MarginalEncoderTrainer`
    and frozen during NRE training.  Everything else (loss, optimiser,
    callbacks, prior truncation, weight transfer across rounds) is inherited
    unchanged from :class:`InferenceNetwork`.

    Activation
    ----------
    Set ``architecture.data_summary.type: "MarginalEncoder"`` in
    ``train_config.yaml``.  ``tmnre.py`` will then call
    :meth:`SequentialTrainer._train_marginal_encoders` before the NRE step
    and instantiate this class instead of :class:`InferenceNetwork`.
    """

    def __init__(
        self,
        train_conf: dict,
        dataset_info: dict,
        normalisation: dict,
        data_summarizer=None,
        periodic_bc_params: list = None,
    ):
        # If loading from checkpoint, data_summarizer is None.
        # Build a dummy MarginalEncoderWrapper with the right architecture so
        # that the parent __init__ succeeds; weights will be overwritten by
        # load_state_dict() afterwards.
        if data_summarizer is None:
            data_summarizer = self._build_dummy_wrapper(train_conf)

        # Parent constructs logratios_model_dict with one MarginalClassifierHead
        # per domain, using n_data_features = bottleneck_dim (shared summary).
        # We override this below with per-marginal n_data_features.
        super().__init__(train_conf, dataset_info, normalisation, data_summarizer, periodic_bc_params)

        # Rebuild logratios_model_dict with per-parameter bottleneck sizes.
        # For a 2D marginal [p, q] the classifier receives two concatenated
        # bottlenecks, so n_data_features = 2 * bottleneck_dim.
        bottleneck_dim = self.data_summary.get_n_features()
        self.logratios_model_dict = nn.ModuleDict()
        for domain in self.marginals_dict_remapped:
            n_data_features_list = [
                len(orig_marg) * bottleneck_dim
                for orig_marg in self.marginals_dict[domain]
            ]
            self.logratios_model_dict[domain] = MarginalClassifierHead(
                n_data_features=n_data_features_list,
                marginals=self.marginals_dict_remapped[domain],
                hlayersizes=train_conf.get("classifier_hlayersizes", (64, 32, 16, 8)),
            ).to(train_conf["device"])

        # Build a flat ordered list of (domain, pos, remapped_indices, original_marginal)
        # so that forward() can pair per-parameter bottlenecks with the right classifier.
        self._marginal_order = []
        for domain in self.marginals_dict_remapped:
            for pos, remapped_marginal in enumerate(self.marginals_dict_remapped[domain]):
                original_marginal = self.marginals_dict[domain][pos]
                self._marginal_order.append((domain, pos, remapped_marginal, original_marginal))

    # ------------------------------------------------------------------
    # Dummy wrapper for checkpoint loading
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dummy_wrapper(train_conf: dict):
        """Create a structurally correct MarginalEncoderWrapper with random
        weights so that the module graph is valid during checkpoint loading.
        Actual weights are restored from the checkpoint state-dict.
        """
        from pembhb.autoencoder import MarginalEncoderTrainer, MarginalEncoderWrapper
        me_conf = train_conf["architecture"]["data_summary"]["MarginalEncoder"]
        marginals_flat = [
            marginal
            for marginal_list in train_conf["marginals"].values()
            for marginal in marginal_list
        ]
        hidden_channels = me_conf.get("hidden_channels", (32, 64, 128, 256, 256))
        if isinstance(hidden_channels, list):
            hidden_channels = tuple(hidden_channels)
        regressor_hidden = me_conf.get("regressor_hidden_sizes", (128, 64))
        if isinstance(regressor_hidden, list):
            regressor_hidden = tuple(regressor_hidden)
        trainer_model = MarginalEncoderTrainer(
            n_channels=me_conf.get("n_channels", 2),
            n_freqs=me_conf.get("n_freqs", 4096),
            marginals=marginals_flat,
            bottleneck_dim=me_conf.get("bottleneck_dim", 200),
            hidden_channels=hidden_channels,
            kernel_size=me_conf.get("kernel_size", 5),
            stride=me_conf.get("stride", 2),
            dropout=me_conf.get("dropout", 0.0),
            residual=me_conf.get("residual", False),
            regressor_hidden_sizes=regressor_hidden,
            representation=me_conf.get("representation", "real_imag"),
        )
        return MarginalEncoderWrapper(trainer_model, freeze=True, device="cpu")

    # ------------------------------------------------------------------
    # Override forward: route per-parameter bottlenecks to classifiers
    # ------------------------------------------------------------------

    def forward(self, d_f, d_t, parameters):
        """Per-parameter forward pass.

        Each parameter's encoder produces an independent bottleneck.
        For 2D marginals, bottlenecks from both parameters are concatenated.
        The output shape is identical to :class:`InferenceNetwork`:
        ``(B, N_marginals)``.
        """
        # bottleneck_dict: {param_idx: (B, bottleneck_dim)}
        bottleneck_dict, _d_t = self.data_summary(d_f, d_t)

        normalised_parameters = (parameters - self.param_mean) / self.param_std
        reparametrised = reparametrise_periodic_bc(normalised_parameters, self.periodic_bc_params)

        logratios_list = []
        for domain, pos, remapped_marginal, original_marginal in self._marginal_order:
            # Concatenate bottlenecks for all params in this marginal
            bottleneck = torch.cat(
                [bottleneck_dict[p] for p in original_marginal], dim=-1
            )  # (B, len(original_marginal) * bottleneck_dim)
            classifier = self.logratios_model_dict[domain].classifiers[pos]
            input_data = torch.cat([bottleneck, reparametrised[:, remapped_marginal]], dim=-1)
            logratios_list.append(classifier(input_data))

        return torch.cat(logratios_list, dim=-1)


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
            # self.fc_blocks.add_module(f"dropout_{i}", nn.Dropout(p=0.2))
            input_size = output_size
        
        self.logratios = MarginalClassifierHead(input_size, marginals=marginals, hidden_size=marginal_hidden_size)
        self.marginals_dict = marginals
        

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
    def __init__(self, n_channels: int , n_timesteps: int, n_freqs: int):
        super().__init__()

        
        
        #self.normalisation_f = nn.BatchNorm1d(num_features=n_channels*2)
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






    def forward(self, d_f, d_t):
        #print(f"d_f shape: {d_f.shape}")
        #d_f_norm = self.normalisation_f(d_f)
        #d_t_norm = self.normalisation_t(d_t)
        #print(f"d_f_norm shape: {d_f_norm.shape}")
        d_f_processed = self.unet_f(d_f)
        d_t_processed = self.unet_t(d_t)
        #print(f"d_f_processed shape: {d_f_processed.shape}")
        flattened_f = self.flatten(d_f_processed)
        flattened_t = self.flatten(d_t_processed)

        #print(f"flattened shape: {flattened.shape}")
        datasummary_fd = self.linear_f(flattened_f)
        datasummary_td = self.linear_t(flattened_t)

        return datasummary_fd, datasummary_td

class ChannelizedLinearCompression(nn.Module): 
    """a linear compression that operates along dimension 1 (channels) independently, and produces, one number for each channel, then concatenates them and passes through a final MLP to get lowdim features.
    """
    def __init__(self, lowdim: int, Nfreqs: int,  in_channels: int  ): 
        """Initializes the ChannelizedLinearCompression module.

        :param lowdim: the output features will be of dimension lowdim
        :type lowdim: int
        :param Nfreqs: the number of frequency bins
        :type Nfreqs: int
        :param in_channels: the number of input channels (note: each TDI channel corresponds to TWO channels in this module, because amplitude and phase are treated separately)
        :type in_channels: int
        """
        super().__init__()
        self.N = Nfreqs
        self.lowdim = lowdim
        self.in_channels = in_channels

        self.channel_blocks = nn.ModuleList()
        int_dimension = (self.N*10)**(0.5)
        for _ in range(self.in_channels):
            block = nn.Sequential(
            nn.Flatten(),  # (B, 1, N) -> (B, N)
            nn.Linear(self.N, int(int_dimension)),
            nn.ReLU(),
            nn.Linear(int(int_dimension), 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # produce a single number per channel
            )
            self.channel_blocks.append(block)

        # final 2-layer MLP: takes concatenated per-channel scalars (B, in_channels) -> lowdim
        final_hidden = 30
        self.final_mlp = nn.Sequential(
            nn.Linear(self.in_channels, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, self.lowdim),
        )
    
    def forward(self, x):
        # x shape: (B, C, N)
        B, C, N = x.shape

        # process each channel with its channel_block -> yields (B,1) per channel
        channel_outputs = []
        for c, block in enumerate(self.channel_blocks):
            out = block(x[:, c, :])  # (B, 1)
            channel_outputs.append(out)

        # concatenate per-channel scalars into (B, C)
        per_channel_scalars = torch.cat(channel_outputs, dim=1)  # (B, C)

        # final MLP to produce lowdim features (B, lowdim)
        compressed = self.final_mlp(per_channel_scalars)  # (B, lowdim)

        return compressed


class BrutalCompression(nn.Module):
    """
    Take top-100 values per channel from d_f and corresponding indices normalized to [0,1].
    Return tensor of shape (B, Nchannels*2, 100) where first Nchannels are values and
    next Nchannels are normalized indices.

    This implementation requires Nfreqs >= 100 and will raise a ValueError otherwise.
    """
    def __init__(self, lowdim: int, Nfreqs: int,  in_channels: int  ): 
        super().__init__()
        self.N = 100  # number of top frequencies to select
        self.lowdim = lowdim
        self.in_channels = in_channels
        self.network = ChannelizedLinearCompression(lowdim, self.N, in_channels)


    def forward(self, d_f, d_t):
        """
        Args:
            d_f: Tensor shape (B, C, Ntot_freqs)
            d_t: passed through unchanged
        Returns:
            compressed: Tensor shape (B, 1, lowdim)
            d_t: unchanged
        """
        B, C, Nfreqs = d_f.shape

        if Nfreqs < self.N:
            raise ValueError(f"BrutalCompression requires at least {self.N} frequencies, got {Nfreqs}")

        k = self.N
        # compute top-k indices from channel 0 only
        _, topk_idx = torch.topk(d_f[:, 0, :].abs(), k=k, dim=1)  # (B, k)
        # expand to all channels for gathering
        indices = topk_idx.unsqueeze(1).expand(-1, C, -1)  # (B, C, k)

        # gather values for all channels at these indices
        selected = torch.gather(d_f, 2, indices)  # (B, C, k)

        compressed = self.network(selected)  # (B, lowdim)

        return compressed, d_t
   
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
        breakpoint()
        x1 = self.up(x1)
        diff_signal_length = x2.size()[2] - x1.size()[2]

        x1 = F.pad(
            x1, [diff_signal_length // 2, diff_signal_length - diff_signal_length // 2]
        )
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


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
        raise NotImplementedError("Unet forward pass is not implemented. The data should be transformed before passing to the Unet.")
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
    

from pembhb.rom import ROMWrapper

DATA_SUMMARY_REGISTRY = { "BrutalCompression": BrutalCompression, "PeregrineModel": PeregrineModel, "ROM": ROMWrapper}


# ---------------------------------------------------------------------------
# Joint Autoencoder + InferenceNetwork
# ---------------------------------------------------------------------------

class SingleGroupReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """ReduceLROnPlateau that only touches a single ``param_group`` of the
    optimizer, and optionally waits until epoch ``>= start_epoch`` before
    doing anything.

    Motivation: ``JointAEInferenceNetwork`` uses one optimizer with two
    param groups (encoder + NRE heads).  Plain ``ReduceLROnPlateau`` reduces
    all groups together and starts counting patience from epoch 0, which
    causes the NRE group's LR to decay during the AE warmup (when
    ``val_nre_loss`` is logged as 0) or because the AE warmup drove
    ``val_loss`` to a minimum the NRE phase can never beat.
    """

    def __init__(self, optimizer, param_group_idx: int = 0,
                 start_epoch: int = 0, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.param_group_idx = param_group_idx
        self.start_epoch = start_epoch
        self._in_warmup = start_epoch > 0

    def step(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if epoch < self.start_epoch:
            self._in_warmup = True
            return

        if self._in_warmup:
            # Entering the active phase: reset tracking so the AE-warmup
            # values don't poison the "best" the NRE phase is compared to.
            self.best = self.mode_worse
            self.num_bad_epochs = 0
            self.cooldown_counter = 0
            self._in_warmup = False

        current = float(metrics)
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        pg = self.optimizer.param_groups[self.param_group_idx]
        old_lr = float(pg["lr"])
        new_lr = max(old_lr * self.factor, self.min_lrs[self.param_group_idx])
        if old_lr - new_lr > self.eps:
            pg["lr"] = new_lr


class JointAEInferenceNetwork(LightningModule):
    """Joint training of an encoder (AE or ME) and NRE classifier heads.

    Supports two encoder types via ``encoder_model``:

    * **DenoisingAutoencoder (AE)** — single encoder trained with MSE
      reconstruction loss; its bottleneck is shared across all NRE heads.
    * **MarginalEncoderTrainer (ME)** — N independent per-parameter encoders
      each trained with MSE regression loss; each NRE head receives only the
      bottleneck(s) for its own parameter(s).

    In both modes the NRE classifier heads receive the bottleneck(s)
    **detached** from the computational graph, so the BCE contrastive loss
    never back-propagates through the encoder.

    An optional warm-up phase trains only the encoder for the first
    ``ae_warmup_epochs`` epochs; the NRE loss is zero during that phase.

    After training, call :meth:`get_encoder_wrapper` to extract the trained
    encoder wrapped for use in a standalone ``InferenceNetwork``.

    Parameters
    ----------
    train_conf : dict
        Full training configuration (same as for ``InferenceNetwork``).
    dataset_info : dict
        YAML sidecar information for the dataset.
    normalisation : dict
        Contains ``td_normalisation``, ``param_mean``, ``param_std``.
    encoder_model : DenoisingAutoencoder | MarginalEncoderTrainer
        An initialised (and normalisation-fitted) encoder.
        Owned by this module.
    ae_warmup_epochs : int
        Number of epochs during which only the encoder loss is active (NRE
        loss is switched off).  Set to 0 to train both from the start.
    lr_ae : float
        Learning rate for the encoder parameter group.
    lr_nre : float
        Learning rate for the NRE classifier heads parameter group.
    ae_weight_decay : float
        Weight decay for the encoder parameter group.
    ae_scheduler_patience : int
        Patience for the encoder learning-rate scheduler.
    ae_scheduler_factor : float
        Factor for the encoder learning-rate scheduler.
    """

    def __init__(
        self,
        train_conf: dict,
        dataset_info: dict,
        normalisation: dict,
        encoder_model,
        ae_warmup_epochs: int = 50,
        lr_ae: float = 1e-3,
        lr_nre: float = 1e-4,
        ae_weight_decay: float = 1e-5,
        ae_scheduler_patience: int = 10,   # deprecated (use ae_scheduler dict)
        ae_scheduler_factor: float = 0.3,  # deprecated (use ae_scheduler dict)
        periodic_bc_params: list = None,
        freeze_ae_after_warmup: bool = False,
        ae_scheduler: dict = None,
        nre_scheduler: dict = None,
    ):
        super().__init__()

        # ---- Encoder (AE or ME) -----------------------------------------
        from pembhb.autoencoder import MarginalEncoderTrainer
        self._is_me = isinstance(encoder_model, MarginalEncoderTrainer)
        self.encoder_model = encoder_model

        # ---- Marginals / NRE setup (mirrors InferenceNetwork) -----------
        self.marginals_dict = train_conf["marginals"]
        self.loss = nn.BCEWithLogitsLoss(reduction="none")
        self.lr_nre = lr_nre
        self.lr_ae = lr_ae
        self.ae_weight_decay = ae_weight_decay
        self.ae_warmup_epochs = ae_warmup_epochs
        self.freeze_ae_after_warmup = freeze_ae_after_warmup

        # Backward-compat: if new dict-style scheduler configs are not
        # provided, derive one from the deprecated scalar args.  Old behaviour
        # used a single scheduler driving both groups on ``val_loss``; we
        # replicate that only when the user has not opted into the new API.
        _legacy = {
            "enabled": True,
            "monitor": "val_loss",
            "mode": "min",
            "factor": ae_scheduler_factor,
            "patience": ae_scheduler_patience,
            "min_lr": 1e-7,
            "start_epoch": 0,
        }
        self.ae_scheduler_config = dict(ae_scheduler) if ae_scheduler is not None else dict(_legacy)
        if nre_scheduler is not None:
            self.nre_scheduler_config = dict(nre_scheduler)
        else:
            # Legacy default: both groups share the same config, but the NRE
            # scheduler is gated to start after the AE warmup so the phase
            # where ``val_nre_loss = 0`` does not corrupt its ``best``.
            self.nre_scheduler_config = dict(_legacy)
            self.nre_scheduler_config["start_epoch"] = ae_warmup_epochs
        # Preserve the deprecated scalars for checkpoint backward-compat.
        self.ae_scheduler_patience = ae_scheduler_patience
        self.ae_scheduler_factor = ae_scheduler_factor

        # Use the actual sampling prior as authoritative source
        _sik = dataset_info.get("sampler_init_kwargs", {})
        if "prior_bounds" in _sik:
            self.bounds_trained = _sik["prior_bounds"]
        else:
            self.bounds_trained = dataset_info["conf"]["prior"]

        self.output_names = []
        self.marginals_list = []
        for domain in self.marginals_dict:
            for marginal in self.marginals_dict[domain]:
                name_output = "_".join(_ORDERED_PRIOR_KEYS[idx] for idx in marginal)
                self.output_names.append(name_output)
                self.marginals_list.append(marginal)

        # ---- Normalisation buffers (same as InferenceNetwork) -----------
        self.td_normalisation = normalisation["td_normalisation"]
        self.register_buffer(
            "param_mean",
            torch.tensor(normalisation["param_mean"], dtype=get_torch_dtype()),
        )
        self.register_buffer(
            "param_std",
            torch.tensor(normalisation["param_std"], dtype=get_torch_dtype()),
        )

        # ---- Periodic BC reparametrisation (mirrors InferenceNetwork) ---
        if periodic_bc_params is None:
            periodic_bc_params = []
        self.periodic_bc_params = periodic_bc_params
        self.param_mean[periodic_bc_params] = 0
        self.param_std[periodic_bc_params] = 1

        # sin/cos normalization statistics (identity fallback for backward compat)
        _n_periodic = len(periodic_bc_params)
        _sc_mean = normalisation.get("sincos_mean", [0.0] * (2 * _n_periodic))
        _sc_std  = normalisation.get("sincos_std",  [1.0] * (2 * _n_periodic))
        self.register_buffer("sincos_mean", torch.tensor(_sc_mean, dtype=get_torch_dtype()))
        self.register_buffer("sincos_std",  torch.tensor(_sc_std,  dtype=get_torch_dtype()))

        self.param_index_remapping = {}
        offset = 0
        for idx in range(len(_ORDERED_PRIOR_KEYS)):
            if idx in self.periodic_bc_params:
                self.param_index_remapping[idx] = [idx + offset, idx + offset + 1]
                offset += 1
            else:
                self.param_index_remapping[idx] = idx + offset

        self.marginals_dict_remapped = {}
        for key in self.marginals_dict.keys():
            marginals_remapped = []
            for marginal in self.marginals_dict[key]:
                current_marginal_remapped = []
                for idx in marginal:
                    if idx in self.periodic_bc_params:
                        current_marginal_remapped.extend(self.param_index_remapping[idx])
                    else:
                        current_marginal_remapped.append(self.param_index_remapping[idx])
                marginals_remapped.append(current_marginal_remapped)
            self.marginals_dict_remapped[key] = marginals_remapped

        # ---- Data summary dimensionality (from encoder) -----------------
        if self._is_me:
            self.n_features_summary = encoder_model.bottleneck_dim
        elif encoder_model.architecture == "conv":
            self.n_features_summary = encoder_model.bottleneck_dim
        else:
            # UNet: need a dummy forward to get bottleneck size
            n_real_ch = encoder_model.n_channels * 2
            with torch.no_grad():
                dummy = torch.zeros(1, n_real_ch, encoder_model.n_freqs)
                b, _ = encoder_model.encoder(dummy)
            self.n_features_summary = b.numel()

        # ---- NRE classifier heads ---------------------------------------
        self.logratios_model_dict = nn.ModuleDict()
        if self._is_me:
            # ME: per-marginal n_data_features (2D marginal gets 2× bottleneck)
            for domain in self.marginals_dict_remapped:
                n_data_features_list = [
                    len(orig_marg) * self.n_features_summary
                    for orig_marg in self.marginals_dict[domain]
                ]
                self.logratios_model_dict[domain] = MarginalClassifierHead(
                    n_data_features=n_data_features_list,
                    marginals=self.marginals_dict_remapped[domain],
                    hlayersizes=train_conf.get("classifier_hlayersizes", (64, 32, 16, 8)),
                )
            # Ordered list for forward(): (domain, pos, remapped_marginal, original_marginal)
            self._marginal_order = []
            for domain in self.marginals_dict_remapped:
                for pos, remapped_marginal in enumerate(self.marginals_dict_remapped[domain]):
                    original_marginal = self.marginals_dict[domain][pos]
                    self._marginal_order.append((domain, pos, remapped_marginal, original_marginal))
        else:
            # AE: shared bottleneck for all marginals in a domain
            for key in self.marginals_dict_remapped:
                self.logratios_model_dict[key] = MarginalClassifierHead(
                    n_data_features=self.n_features_summary * len(key),
                    marginals=self.marginals_dict_remapped[key],
                    hlayersizes=train_conf.get("classifier_hlayersizes", (64, 32, 16, 8)),
                )

        # ---- Save hyper-parameters for checkpoint / utils compat --------
        self.save_hyperparameters(
            {
                "train_conf": train_conf,
                "dataset_info": dataset_info,
                "normalisation": {
                    k: v.tolist() if hasattr(v, "tolist") else v
                    for k, v in normalisation.items()
                },
                "ae_warmup_epochs": ae_warmup_epochs,
                "lr_ae": lr_ae,
                "lr_nre": lr_nre,
                "ae_weight_decay": ae_weight_decay,
                # Deprecated scalars kept for backward-compat with older ckpts
                "ae_scheduler_patience": ae_scheduler_patience,
                "ae_scheduler_factor": ae_scheduler_factor,
                # New per-group scheduler configs (authoritative)
                "ae_scheduler": self.ae_scheduler_config,
                "nre_scheduler": self.nre_scheduler_config,
            },
            logger=True,
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Backward compatibility: old checkpoints don't have sincos_mean/sincos_std
        # buffers. Fall back to the identity-normalization values built in __init__
        # (mean=0, std=1) so load_state_dict doesn't fail on missing keys.
        for buf_name in ("sincos_mean", "sincos_std"):
            key = prefix + buf_name
            if key not in state_dict and hasattr(self, buf_name):
                state_dict[key] = getattr(self, buf_name)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    @property
    def autoencoder(self):
        """Backward-compatibility alias for ``self.encoder_model``."""
        return self.encoder_model

    # ------------------------------------------------------------------
    # Forward  (NRE path — used at inference / posterior evaluation time)
    # ------------------------------------------------------------------

    def _encode_detached(self, d_f: torch.Tensor):
        """Encode FD data with no NRE gradient flow.

        Returns a detached ``(B, bottleneck_dim)`` tensor for AE mode, or a
        detached ``{param_idx: (B, bottleneck_dim)}`` dict for ME mode.
        """
        if self._is_me:
            x_norm = self.encoder_model.preprocess(d_f)
            return {
                p: enc(x_norm).detach()
                for p, enc in zip(self.encoder_model.param_indices, self.encoder_model.encoders)
            }
        else:
            x_norm = self.encoder_model.preprocess(d_f)
            bottleneck = self.encoder_model.encode(x_norm)
            if self.encoder_model.architecture != "conv":
                bottleneck, _ = bottleneck
                bottleneck = bottleneck.reshape(bottleneck.shape[0], -1)
            return bottleneck.detach()

    def _encode(self, d_f: torch.Tensor):
        """Encode FD data with gradients (used only for encoder loss)."""
        if self._is_me:
            x_norm = self.encoder_model.preprocess(d_f)
            return {
                p: enc(x_norm)
                for p, enc in zip(self.encoder_model.param_indices, self.encoder_model.encoders)
            }
        else:
            x_norm = self.encoder_model.preprocess(d_f)
            bottleneck = self.encoder_model.encode(x_norm)
            if self.encoder_model.architecture != "conv":
                bottleneck, _ = bottleneck
                bottleneck = bottleneck.reshape(bottleneck.shape[0], -1)
            return bottleneck

    def forward(self, d_f, d_t, parameters):
        """NRE forward pass (same signature as InferenceNetwork.forward).

        Used by ``utils.get_logratios_grid`` and posterior evaluation.
        Bottleneck(s) are detached so this is safe inside a training loop.
        """
        normalised_parameters = (parameters - self.param_mean) / self.param_std
        reparametrised_withbc_params = reparametrise_periodic_bc(normalised_parameters, self.periodic_bc_params)
        if len(self.periodic_bc_params) > 0:
            reparametrised_withbc_params = normalise_sincos_cols(
                reparametrised_withbc_params, self.periodic_bc_params,
                self.param_index_remapping, self.sincos_mean, self.sincos_std,
            )

        if self._is_me:
            bottleneck_dict = self._encode_detached(d_f)
            logratios_list = []
            for domain, pos, remapped_marginal, original_marginal in self._marginal_order:
                bottleneck = torch.cat(
                    [bottleneck_dict[p] for p in original_marginal], dim=-1
                )
                classifier = self.logratios_model_dict[domain].classifiers[pos]
                input_data = torch.cat([bottleneck, reparametrised_withbc_params[:, remapped_marginal]], dim=-1)
                logratios_list.append(classifier(input_data))
        else:
            bottleneck = self._encode_detached(d_f)
            features_dict = {"ft": None, "f": bottleneck, "t": d_t}
            logratios_list = []
            for key in self.logratios_model_dict.keys():
                logratios_list.append(
                    self.logratios_model_dict[key](features_dict[key], reparametrised_withbc_params)
                )
        return torch.cat(logratios_list, dim=-1)

    # ------------------------------------------------------------------
    # NRE loss (contrastive BCE — mirrors InferenceNetwork)
    # ------------------------------------------------------------------

    def _calc_nre_loss(self, batch):
        """Compute the contrastive NRE loss and logits."""
        data_f = batch["wave_fd"] + batch["noise_fd"]
        if "wave_td" in batch:
            data_t = batch["wave_td"] + batch["noise_td"]
            all_data_t = torch.cat((data_t, data_t), dim=0)
        else:
            all_data_t = None
        parameters = batch["source_parameters"]

        all_data_f = torch.cat((data_f, data_f), dim=0)
        scrambled_params = torch.roll(parameters, shifts=1, dims=0)
        all_params = torch.cat((parameters, scrambled_params), dim=0)

        all_logits = self(all_data_f, all_data_t, all_params)

        shape_half = (all_logits.shape[0] // 2, all_logits.shape[1])
        labels = torch.cat(
            (
                torch.ones(shape_half, device=self.device, dtype=get_torch_dtype()),
                torch.zeros(shape_half, device=self.device, dtype=get_torch_dtype()),
            ),
            dim=0,
        )
        all_loss = self.loss(all_logits, labels)
        task_losses = torch.mean(all_loss, dim=0)
        nre_loss = torch.mean(task_losses)
        return all_logits, task_losses, nre_loss

    # ------------------------------------------------------------------
    # Encoder loss (AE reconstruction MSE or ME regression MSE)
    # ------------------------------------------------------------------

    def _calc_encoder_loss(self, batch):
        """Compute the encoder training loss.

        AE mode: standard MSE reconstruction loss on normalised representations.
        (Noise-weighted MSE caused bottleneck collapse with LISA ASD dynamic range.)

        ME mode: sum of per-parameter MSE regression losses on normalised targets.
        """
        if self._is_me:
            noisy = batch["wave_fd"] + batch["noise_fd"]
            params = batch["source_parameters"]
            x_norm = self.encoder_model.preprocess(noisy)
            total_loss = torch.tensor(0.0, device=self.device, dtype=x_norm.dtype)
            for encoder, regressor, param_idx in zip(
                self.encoder_model.encoders,
                self.encoder_model.regressors,
                self.encoder_model.param_indices,
            ):
                bottleneck = encoder(x_norm)
                predicted = regressor(bottleneck)                        # (B, 1)
                target_raw = params[:, param_idx:param_idx + 1].to(x_norm.dtype)
                target_norm = (
                    (target_raw - self.encoder_model.param_mean[param_idx])
                    / (self.encoder_model.param_std[param_idx] + 1e-30)
                )
                total_loss = total_loss + F.mse_loss(predicted, target_norm)
            return total_loss
        else:
            noisy = batch["wave_fd"] + batch["noise_fd"]
            clean = batch["wave_fd"]
            noisy_norm = self.encoder_model.preprocess(noisy)
            clean_norm = self.encoder_model.preprocess(clean)
            reconstructed = self.encoder_model(noisy_norm)
            target = self.encoder_model._get_target(clean_norm)
            # --- TEMP: target-scale check. Comment out once verified. ---
            if not getattr(self, "_target_scale_printed", False):
                print(
                    f"[AE-scale] whiten={self.encoder_model.whiten}  "
                    f"target.abs().mean()={target.abs().mean().item():.3e}  "
                    f"target.abs().max()={target.abs().max().item():.3e}  "
                    f"noisy.abs().mean(complex)={noisy.abs().mean().item():.3e}",
                    flush=True,
                )
                self._target_scale_printed = True
            loss =  F.mse_loss(reconstructed, target)
            return loss

    def _calc_ae_loss(self, batch):
        """Deprecated alias for ``_calc_encoder_loss``."""
        return self._calc_encoder_loss(batch)

    # ------------------------------------------------------------------
    # Accuracy (same as InferenceNetwork)
    # ------------------------------------------------------------------

    def _calc_accuracies(self, all_logits):
        shape_half = (all_logits.shape[0] // 2, all_logits.shape[1])
        joint_preds = (all_logits[: shape_half[0]] > 0).float()
        scrambled_preds = (all_logits[shape_half[0] :] > 0).float()
        joint_accurate = (joint_preds == 1).float()
        scrambled_accurate = (scrambled_preds == 0).float()
        joint_accuracy = torch.mean(joint_accurate, dim=0)
        scrambled_accuracy = torch.mean(scrambled_accurate, dim=0)
        accuracy_params = (joint_accuracy + scrambled_accuracy) / 2
        accuracy = torch.mean(accuracy_params)
        return accuracy_params, accuracy

    # ------------------------------------------------------------------
    # Lightning training / validation steps
    # ------------------------------------------------------------------

    def on_train_epoch_start(self):
        if self.freeze_ae_after_warmup and self.current_epoch == self.ae_warmup_epochs:
            for param in self.encoder_model.parameters():
                param.requires_grad_(False)
            print(f"[Joint] Encoder frozen at epoch {self.current_epoch} "
                  f"(ae_warmup_epochs={self.ae_warmup_epochs})")

    def training_step(self, batch, batch_idx):
        enc_frozen = self.freeze_ae_after_warmup and self.current_epoch >= self.ae_warmup_epochs
        encoder_loss = torch.tensor(0.0, device=self.device) if enc_frozen else self._calc_encoder_loss(batch)

        in_warmup = self.current_epoch < self.ae_warmup_epochs
        if in_warmup:
            nre_loss = torch.tensor(0.0, device=self.device)
            all_logits = None
            task_losses = None
        else:
            all_logits, task_losses, nre_loss = self._calc_nre_loss(batch)

        total_loss = encoder_loss + nre_loss

        enc_log_key = "train_me_loss" if self._is_me else "train_ae_loss"
        self.log(enc_log_key, encoder_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_nre_loss", nre_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if in_warmup:
            self.log("train_accuracy", 0.5, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            accuracy_params, accuracy = self._calc_accuracies(all_logits)
            self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            overall_idx = 0
            for domain in self.marginals_dict:
                for i, marginal in enumerate(self.marginals_dict[domain]):
                    name = self.output_names[overall_idx]
                    self.log(f"train_accuracy_{name}", accuracy_params[overall_idx], on_step=True, on_epoch=True, logger=True)
                    self.log(f"train_loss_{name}", task_losses[overall_idx], on_step=True, on_epoch=True, logger=True)
                    overall_idx += 1

        return total_loss

    def validation_step(self, batch, batch_idx):
        enc_frozen = self.freeze_ae_after_warmup and self.current_epoch >= self.ae_warmup_epochs
        encoder_loss = torch.tensor(0.0, device=self.device) if enc_frozen else self._calc_encoder_loss(batch)

        in_warmup = self.current_epoch < self.ae_warmup_epochs
        if in_warmup:
            nre_loss = torch.tensor(0.0, device=self.device)
            all_logits = None
            task_losses = None
        else:
            all_logits, task_losses, nre_loss = self._calc_nre_loss(batch)

        total_loss = encoder_loss + nre_loss

        enc_log_key = "val_me_loss" if self._is_me else "val_ae_loss"
        self.log(enc_log_key, encoder_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_nre_loss", nre_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if in_warmup:
            self.log("val_accuracy", 0.5, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            accuracy_params, accuracy = self._calc_accuracies(all_logits)
            self.log("val_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            overall_idx = 0
            for domain in self.marginals_dict:
                for marginal in self.marginals_dict[domain]:
                    name = self.output_names[overall_idx]
                    self.log(f"val_accuracy_{name}", accuracy_params[overall_idx], on_step=False, on_epoch=True, logger=True)
                    self.log(f"val_loss_{name}", task_losses[overall_idx], on_step=False, on_epoch=True, logger=True)
                    overall_idx += 1

        return total_loss

    # ------------------------------------------------------------------
    # Optimizers (two parameter groups, single optimizer)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.encoder_model.parameters(),
                    "lr": self.lr_ae,
                    "weight_decay": self.ae_weight_decay,
                },
                {
                    "params": self.logratios_model_dict.parameters(),
                    "lr": self.lr_nre,
                },
            ],
        )

        scheduler_configs = []
        for cfg, group_idx, label in (
            (self.ae_scheduler_config, 0, "ae"),
            (self.nre_scheduler_config, 1, "nre"),
        ):
            if not cfg.get("enabled", True):
                continue
            sched = SingleGroupReduceLROnPlateau(
                optimizer,
                param_group_idx=group_idx,
                start_epoch=cfg.get("start_epoch", 0),
                mode=cfg.get("mode", "min"),
                factor=cfg.get("factor", 0.3),
                patience=cfg.get("patience", 10),
                min_lr=cfg.get("min_lr", 1e-7),
            )
            scheduler_configs.append({
                "scheduler": sched,
                "monitor": cfg.get("monitor", "val_loss"),
                "interval": "epoch",
                "frequency": 1,
                "name": f"lr-{label}",
            })

        if not scheduler_configs:
            return optimizer
        # Multiple schedulers attached to a single optimizer: Lightning expects
        # ([optimizer], [lr_scheduler_config, ...]).
        return [optimizer], scheduler_configs

    # ------------------------------------------------------------------
    # Helpers (for compatibility with existing utils / callbacks)
    # ------------------------------------------------------------------

    def get_encoder_wrapper(self, freeze: bool = True):
        """Extract the trained encoder wrapped for standalone inference.

        Returns an ``AutoencoderWrapper`` (AE mode) or a
        ``MarginalEncoderWrapper`` (ME mode).
        """
        if self._is_me:
            from pembhb.autoencoder import MarginalEncoderWrapper
            return MarginalEncoderWrapper(self.encoder_model, freeze=freeze)
        else:
            from pembhb.autoencoder import AutoencoderWrapper
            return AutoencoderWrapper(self.encoder_model, freeze=freeze)

    def get_autoencoder_wrapper(self, freeze: bool = True):
        """Deprecated alias for :meth:`get_encoder_wrapper`."""
        return self.get_encoder_wrapper(freeze=freeze)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None, **kwargs):
        """Load a ``JointAEInferenceNetwork`` from a Lightning checkpoint.

        ``encoder_model`` is not stored in hparams, so the standard Lightning
        ``load_from_checkpoint`` would fail.  This override reconstructs the
        encoder architecture from ``train_conf`` stored in the checkpoint.

        Supports both AE (``DenoisingAutoencoder``) and ME
        (``MarginalEncoderTrainer``) encoder types, detected from
        ``train_conf["architecture"]["data_summary"]["type"]``.

        Also handles old-style checkpoints whose state-dict uses the legacy
        ``"autoencoder.*"`` key prefix by remapping it to ``"encoder_model.*"``.
        """
        from pembhb.autoencoder import (
            DenoisingAutoencoder, MarginalEncoderTrainer,
        )
        device = map_location or ("cuda" if torch.cuda.is_available() else "cpu")

        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        hp = raw["hyper_parameters"]
        ds_type = (
            hp["train_conf"].get("architecture", {})
            .get("data_summary", {})
            .get("type", "Autoencoder")
        )

        # Remap old-style "autoencoder.*" keys → "encoder_model.*"
        state_dict = raw["state_dict"]
        if any(k.startswith("autoencoder.") for k in state_dict):
            state_dict = {
                ("encoder_model." + k[len("autoencoder."):] if k.startswith("autoencoder.") else k): v
                for k, v in state_dict.items()
            }

        if ds_type == "MarginalEncoder":
            me_conf = hp["train_conf"]["architecture"]["data_summary"]["MarginalEncoder"]
            marginals_flat = [
                m for mlist in hp["train_conf"]["marginals"].values() for m in mlist
            ]
            hidden_channels = me_conf.get("hidden_channels", (32, 64, 128, 256, 256))
            if isinstance(hidden_channels, list):
                hidden_channels = tuple(hidden_channels)
            regressor_hidden = me_conf.get("regressor_hidden_sizes", (128, 64))
            if isinstance(regressor_hidden, list):
                regressor_hidden = tuple(regressor_hidden)
            # Infer residual from state-dict keys (config value may be stale)
            me_keys = [k for k in state_dict if k.startswith("encoder_model.encoders")]
            residual = any(".main." in k for k in me_keys)
            dummy_encoder = MarginalEncoderTrainer(
                n_channels=me_conf.get("n_channels", 2),
                n_freqs=me_conf.get("n_freqs", 4096),
                marginals=marginals_flat,
                bottleneck_dim=me_conf.get("bottleneck_dim", 200),
                hidden_channels=hidden_channels,
                kernel_size=me_conf.get("kernel_size", 5),
                stride=me_conf.get("stride", 2),
                dropout=me_conf.get("dropout", 0.0),
                residual=residual,
                regressor_hidden_sizes=regressor_hidden,
                representation=me_conf.get("representation", "real_imag"),
            )
        else:  # "Autoencoder"
            ae_conf = hp["train_conf"]["architecture"]["data_summary"]["Autoencoder"]
            ae_keys = [k for k in state_dict if k.startswith("encoder_model.encoder")]
            residual = any(".main." in k for k in ae_keys)
            # Infer decoder_post_fc_bn from state-dict keys (config value may
            # be stale, e.g. when reloading an older checkpoint after toggling
            # the flag in train_config.yaml).  BatchNorm1d registers
            # post_fc_norm.weight; nn.Identity registers nothing.
            decoder_post_fc_bn = any(
                k.endswith("decoder.post_fc_norm.weight") for k in state_dict
            )
            dummy_encoder = DenoisingAutoencoder(
                n_channels=ae_conf["n_channels"],
                n_freqs=ae_conf["n_freqs"],
                architecture=ae_conf.get("architecture", "conv"),
                bottleneck_dim=ae_conf["bottleneck_dim"],
                hidden_channels=ae_conf["hidden_channels"],
                kernel_size=ae_conf["kernel_size"],
                stride=ae_conf["stride"],
                dropout=ae_conf.get("dropout", 0.0),
                residual=residual,
                decoder_post_fc_bn=decoder_post_fc_bn,
                representation=ae_conf.get("representation", "real_imag"),
                high_freq_only=ae_conf.get("high_freq_only", False),
                freq_split_idx=ae_conf.get("freq_split_idx", 2048),
                idx_lowerbound=ae_conf.get("idx_lowerbound", None),
                idx_upperbound=ae_conf.get("idx_upperbound", None),
                whiten=ae_conf.get("whiten", True),
                amplitude_normalise=ae_conf.get("amplitude_normalise", False),
                subtract_mean_whitened=ae_conf.get("subtract_mean_whitened", False)
            )

        norm = hp["normalisation"]
        model = cls(
            train_conf=hp["train_conf"],
            dataset_info=hp["dataset_info"],
            normalisation=norm,
            encoder_model=dummy_encoder,
            ae_warmup_epochs=hp.get("ae_warmup_epochs", 0),
            lr_ae=hp.get("lr_ae", 1e-3),
            lr_nre=hp.get("lr_nre", 1e-4),
            ae_weight_decay=hp.get("ae_weight_decay", 0.0),
            ae_scheduler_patience=hp.get("ae_scheduler_patience", 10),
            ae_scheduler_factor=hp.get("ae_scheduler_factor", 0.3),
            ae_scheduler=hp.get("ae_scheduler"),
            nre_scheduler=hp.get("nre_scheduler"),
            periodic_bc_params=hp["train_conf"].get("periodic_bc_params"),
        )
        # i added a branch so that the whiten:False restores a behaviour where
        # mean_vec and global_scale_factor are used. 
        # these buffers are now registered in the model by default, but old checkpoints dont store them. 
        # hence, we need to ignore their absence when loading old checkpoints. 
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Old checkpoints predate the baseline-pipeline buffers; their identity-init
        # defaults (mean_vec=0, global_scale_factor=1) are correct for whiten=True.
        allowed_missing = {"encoder_model.mean_vec", "encoder_model.global_scale_factor", "encoder_model.mean_whitened"}
        unexpected_real = set(unexpected)
        missing_real = set(missing) - allowed_missing
        if missing_real or unexpected_real:
            raise RuntimeError(
                f"Unexpected state_dict mismatch. missing={missing_real}, "
                f"unexpected={unexpected_real}"
            )        
        model.to(device)
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Checkpoint-aware model loader
# ---------------------------------------------------------------------------

# Registry: data_summary type string → model class.
# Add entries here when new InferenceNetwork subclasses are introduced.
_SUMMARY_TYPE_TO_MODEL_CLASS: dict = {
    "MarginalEncoder": PerMarginalInferenceNetwork,
}


def load_inference_network(ckpt_path: str, device: str = None) -> InferenceNetwork:
    """Load an InferenceNetwork (or subclass) from a Lightning checkpoint.

    Inspects ``hparams["train_conf"]["architecture"]["data_summary"]["type"]``
    to determine the correct class before calling ``load_from_checkpoint``,
    so the right ``forward`` implementation is always used.

    Parameters
    ----------
    ckpt_path : str
        Path to a ``*.ckpt`` file.
    device : str, optional
        Target device string (e.g. ``"cuda"`` or ``"cpu"``).  Defaults to
        ``"cuda"`` when a GPU is available, otherwise ``"cpu"``.

    Returns
    -------
    InferenceNetwork
        The loaded model in eval mode on *device*.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Peek at hparams without constructing the model.
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = raw.get("hyper_parameters", {})

    # Joint checkpoints (JointAEInferenceNetwork) carry ae_warmup_epochs in
    # their hparams; sequential checkpoints do not.
    if "ae_warmup_epochs" in hp:
        cls = JointAEInferenceNetwork
    else:
        ds_type = (
            hp.get("train_conf", {})
              .get("architecture", {})
              .get("data_summary", {})
              .get("type", None)
        )
        cls = _SUMMARY_TYPE_TO_MODEL_CLASS.get(ds_type, InferenceNetwork)

    model = cls.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    return model