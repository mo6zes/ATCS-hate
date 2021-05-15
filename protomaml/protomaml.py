import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import transformers
import pytorch_lightning as pl
import torchmetrics.functional as f
from copy import deepcopy
from collections import defaultdict

from models.BERT import BERT
from .utils import step

higher.optim.DifferentiableOptimizer.step = step


class ProtoMAML(pl.LightningModule):
    def __init__(self, model='bert-base-uncased', hidden_size=512, output_size=512, gradient_checkpointing=False, inner_updates=100, lr=5e-5, weight_decay=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        
        self.model = BERT(model, hidden_size=hidden_size, output_size=output_size, gradient_checkpointing=gradient_checkpointing)
        self.model.unfreeze_attention_layer([6, 7, 8, 9, 10, 11])
#         self.model.unfreeze_module(nn.LayerNorm)
        self.model.train()
        
        # protolayer weight and bias
        self.weight = 0 
        self.bias = 0
        self.output_lr = self.hparams.lr * 100
        
        # metric logging
        self.metric_dict = defaultdict(lambda: defaultdict(list))
        
    def feature_forward(self, x):
        return self.model(x)
    
    def forward(self, x):
        x = self.feature_forward(x)
        x = self.protolayer(x)
        return x

    def training_step(self, batch, batch_indx):
        if not self.hparams.alt_step:
            loss = self.default_step(batch, batch_indx)
        else:
            loss = self.alt_step(batch, batch_indx)
        
    def default_step(self, batch, batch_indx):
        # get the optimizer
        opt = self.optimizers()
        
        # loop over sampled tasks
        for task in batch:
            # get the support and query dataloader
            support_loader = task.support_loader
            query_loader = task.query_loader
            
            # create prototype layer
            weight, bias = self.calculate_prototypes(support_loader, task.n_classes)
            
            # perform the inner loops
            # clone the model and create a differentiable optimizer
            # higher creates a differentiable inner loop for us
            with higher.innerloop_ctx(self.model, opt.optimizer, copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
                # ensure that our model is trainable and reset the gradients.
                self.train()
                fmodel.train()
                self.zero_grad(fmodel)
                
                # perform task adaptation for k inner steps
                data_iter = iter(support_loader)
                for i in range(min(self.hparams.inner_updates, len(support_loader))):
                    batch_x, batch_y = next(data_iter)
                    batch_x = [i.to(self.device) for i in batch_x]
                    batch_y = batch_y.to(self.device)
                    pred_y = self.protolayer(fmodel(batch_x), weight, bias)
                    support_loss = F.cross_entropy(pred_y, batch_y)
                    
                    weight_grad, bias_grad = torch.autograd.grad(support_loss, [self.weight, self.bias], retain_graph=True)
                    self.weight = self.weight - self.output_lr * weight_grad
                    self.bias = self.bias - self.output_lr * bias_grad
                    
                    diffopt.step(support_loss, retain_graph=True)
                    del support_loss
                    
                    self.calc_metrics(pred_y, batch_y, task.n_classes, mode='support')
                    del pred_y

                # subsitute the orgininal prototypes back in the grad graph.
                # this might be unnesesairy, but i'm unsure
                self.weight = torch.zeros_like(weight, requires_grad=True) + (self.weight - torch.zeros_like(weight, requires_grad=True)).detach()
                self.bias = torch.zeros_like(bias, requires_grad=True) + (self.bias - torch.zeros_like(bias, requires_grad=True)).detach()

                # abtain the gradient on the query set
                for module in fmodel.modules():
                    if isinstance(module, nn.Dropout):
                        module.eval()
    
                for i, (batch_x, batch_y) in enumerate(query_loader):
                    batch_x = [i.to(self.device) for i in batch_x]
                    batch_y = batch_y.to(self.device)
                    pred_y = self.protolayer(fmodel(batch_x), weight, bias)
                    query_loss = F.cross_entropy(pred_y, batch_y)
                    self.calc_metrics(pred_y, batch_y, task.n_classes, mode='query')
                    
                    # calculate the gradients
                    # set create_graph=True for higher order derivatives
                    grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, fmodel.parameters()), retain_graph=True, create_graph=False)
                    if i == len(query_loader)-1:
                        meta_grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, self.model.parameters()), retain_graph=False, create_graph=False)
                    else:
                        meta_grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, self.model.parameters()), retain_graph=True, create_graph=False)
                    del query_loss
                    del pred_y
#                     torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, fmodel.parameters()), self.hparams.grad_clip)
#                     torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.hparams.grad_clip)
                    
                    # save the gradients in the model
                    for param, grad, meta_grad in zip(filter(lambda p: p.requires_grad, self.model.parameters()), grads, meta_grads):
                        if param.grad is not None:
                            param.grad += grad + meta_grad
                        else:
                            param.grad = grad + meta_grad
        
        # update the original model parameters and reset the gradients
        opt.step()
        self.zero_grad(self.model)
        
        metrics = self.reduce_metrics('train')
        
        self.log("s_loss", metrics['support_loss'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log("s_acc", metrics['support_acc'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log("q_loss", metrics['query_loss'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log("q_acc", metrics['query_acc'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        
        return metrics['query_loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.bert.base_model.parameters())},
                                      {'params': filter(lambda p: p.requires_grad, self.model.mlp.parameters()), 'lr': self.hparams.lr*100}],
                                      lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=5)
        return [optimizer], [lr_scheduler]
    
    def init_prototypes(self, model_output, labels, n_classes):
        prototypes = torch.zeros((n_classes, model_output.shape[-1]), device=self.device)
        for c in range(n_classes):
            indices = torch.nonzero(labels == c).view(-1)
            if indices.numel() != 0:
                prototypes[c] = torch.mean(model_output.index_select(0, indices), dim=0)
        self.prototypes = prototypes
        weight = 2 * self.prototypes
        bias = -(self.prototypes.norm(dim=-1)**2)
        self.weight = torch.zeros_like(weight, requires_grad=True)
        self.bias = torch.zeros_like(bias, requires_grad=True)
        return weight, bias
    
    def calculate_prototypes(self, support_loader, n_classes):
        loader = iter(support_loader)
        res = []
        label = []
        for batch_x, batch_y in loader:
            batch_x = [i.to(self.device) for i in batch_x]
            batch_y = batch_y.to(self.device)
            pred_y = self.model(batch_x)
            res.append(pred_y)
            label.append(batch_y)
        weight, bias = self.init_prototypes(torch.cat(res, dim=0), torch.cat(label, dim=0), n_classes)
        del res
        del label
        del loader
        return weight, bias
        
    def protolayer(self, tensor, weight, bias):
        return F.linear(tensor, self.weight+weight, self.bias+bias)
        
    def zero_grad(self, module):
        for param in filter(lambda p: p.requires_grad, module.parameters()):
            param.grad = None
            
    def calc_metrics(self, pred_y, batch_y, num_classes, mode='support'):
        self.metric_dict[mode]['loss'].append(F.cross_entropy(pred_y.detach(), batch_y.detach()))
        self.metric_dict[mode]['acc'].append(f.accuracy(F.softmax(pred_y.detach(), dim=-1), batch_y.detach()))
        self.metric_dict[mode]['f1'].append(f.f1(F.softmax(pred_y.detach(), dim=-1), batch_y.detach(), num_classes=num_classes))
        
    def reduce_metrics(self, state='train', prog_bar_metrics=['loss', 'acc']):
        metrics = {}
        for mode in self.metric_dict.keys():
            for metric in self.metric_dict[mode].keys():
                mean_m = torch.stack(self.metric_dict[mode][metric]).mean()
                if metric in prog_bar_metrics and state == 'train':
                    metrics[mode+'_'+metric] = mean_m
                self.log(f"{'_'.join([state, mode, metric])}", mean_m, on_step=False, on_epoch=True)
        self.metric_dict = defaultdict(lambda: defaultdict(list))
        return metrics
    
    
    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        loss = self.eval_step(batch, batch_idx, mode="val")
        torch.set_grad_enabled(False)
        return loss
    
    def test_step(self, batch, batch_indx):
        torch.set_grad_enabled(True)
        loss = self.eval_step(batch, batch_idx, mode="test")
        torch.set_grad_enabled(False)
        return loss
    
    def eval_step(self, batch, batch_indx, mode="val"):
        opt = self.optimizers()
        
        for task in batch:
            support_loader = task.support_loader
            query_loader = task.query_loader
            
            weight, bias = self.calculate_prototypes(support_loader, task.n_classes)
            
            with higher.innerloop_ctx(self.model, opt.optimizer, copy_initial_weights=False, track_higher_grads=False) as (fmodel, diffopt):
                fmodel.train()
                self.zero_grad(fmodel)
                
                # perform task adaptation for k inner steps
                data_iter = iter(support_loader)
                for i in range(min(self.hparams.eval_updates, len(support_loader))):
                    batch_x, batch_y = next(data_iter)
                    batch_x = [i.to(self.device) for i in batch_x]
                    batch_y = batch_y.to(self.device)
                    pred_y = self.protolayer(fmodel(batch_x), weight, bias)
                    support_loss = F.cross_entropy(pred_y, batch_y)
                    
                    weight_grad, bias_grad = torch.autograd.grad(support_loss, [self.weight, self.bias], retain_graph=True)
                    self.weight = self.weight - self.output_lr * weight_grad
                    self.bias = self.bias - self.output_lr * bias_grad
                    
                    diffopt.step(support_loss, retain_graph=True)
                    del support_loss
                    
                    self.calc_metrics(pred_y, batch_y, task.n_classes, mode='support')
                    del pred_y

                # test on the query set
                fmodel.eval()
                with torch.nograd():
                    for i, (batch_x, batch_y) in enumerate(query_loader):
                        batch_x = [i.to(self.device) for i in batch_x]
                        batch_y = batch_y.to(self.device)
                        pred_y = self.protolayer(fmodel(batch_x), weight, bias)
                        query_loss = F.cross_entropy(pred_y, batch_y)
                        self.calc_metrics(pred_y, batch_y, task.n_classes, mode='query')
        
        metrics = self.reduce_metrics(mode)
        
        return metrics['query_loss']
            
    def alt_step(self, batch, batch_indx):
        # get the optimizer
        opt = self.optimizers()
        
        # loop over sampled tasks
        for task in batch:
            # get the support and query dataloader
            support_loader = task.support_loader
            query_loader = task.query_loader
            
            model_e = deepcopy(self.model)
            
            # create prototype layer
            weight, bias = self.calculate_prototypes(support_loader, task.n_classes)
            
            e_opt = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model_e.bert.base_model.parameters())},
                                      {'params': filter(lambda p: p.requires_grad, model_e.mlp.parameters()), 'lr': self.hparams.lr*100},
                                      {'params': [self.weight, self.bias], 'lr': self.hparams.lr*100}],
                                      lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            
            # ensure that our model is trainable and reset the gradients.
            self.train()
            model_e.train()
            self.zero_grad(model_e)

            # perform task adaptation for k inner steps
            data_iter = iter(support_loader)
            for i in range(min(self.hparams.inner_updates, len(support_loader))):
                batch_x, batch_y = next(data_iter)
                batch_x = [i.to(self.device) for i in batch_x]
                batch_y = batch_y.to(self.device)
                
                pred_y = self.protolayer(model_e(batch_x), weight, bias)
                support_loss = F.cross_entropy(pred_y, batch_y)
                                           
                weight_grad, bias_grad = torch.autograd.grad(support_loss, [self.weight, self.bias], retain_graph=True)
                if self.weight.grad is not None:
                    self.weight.grad += weight_grad
                else:
                    self.weight.grad = weight_grad
                if self.bias.grad is not None:
                    self.bias.grad += bias_grad
                else:
                    self.bias.grad = bias_grad

                grads = torch.autograd.grad(support_loss, filter(lambda p: p.requires_grad, model_e.parameters()), retain_graph=False)
                for param, grad in zip(filter(lambda p: p.requires_grad, model_e.parameters()), grads):
                    if param.grad is not None:
                        param.grad += grad
                    else:
                        param.grad = grad
                
                e_opt.step()
                self.zero_grad(model_e)
                self.weight.grad = None
                self.bias.grad = None

                self.calc_metrics(pred_y, batch_y, task.n_classes, mode='support')

            # subsitute the orgininal prototypes back in the grad graph.
            # this might be unnesesairy, but i'm unsure
            self.weight = torch.zeros_like(weight, requires_grad=True) + (self.weight - torch.zeros_like(weight, requires_grad=True)).detach()
            self.bias = torch.zeros_like(bias, requires_grad=True) + (self.bias - torch.zeros_like(bias, requires_grad=True)).detach()

            # abtain the gradient on the query set
            for module in model_e.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

            for i, (batch_x, batch_y) in enumerate(query_loader):
                batch_x = [i.to(self.device) for i in batch_x]
                batch_y = batch_y.to(self.device)
                pred_y = self.protolayer(model_e(batch_x), weight, bias)
                query_loss = F.cross_entropy(pred_y, batch_y)
                
                self.calc_metrics(pred_y, batch_y, task.n_classes, mode='query')

                # calculate the gradients
                # set create_graph=True for higher order derivatives
                grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, model_e.parameters()), retain_graph=True)
                if i == len(query_loader)-1:
                    meta_grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, self.model.parameters()), retain_graph=False)
                else:
                    meta_grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, self.model.parameters()), retain_graph=True)

#                     torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, fmodel.parameters()), self.hparams.grad_clip)
#                     torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), self.hparams.grad_clip)

                # save the gradients in the model
                for param, grad, meta_grad in zip(filter(lambda p: p.requires_grad, self.model.parameters()), grads, meta_grads):
                    if param.grad is not None:
                        param.grad += grad + meta_grad
                    else:
                        param.grad = grad + meta_grad
        
        # update the original model parameters and reset the gradients
        opt.step()
        self.zero_grad(self.model)

        metrics = self.reduce_metrics('train')
        
        self.log("s_loss", metrics['support_loss'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log("s_acc", metrics['support_acc'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log("q_loss", metrics['query_loss'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        self.log("q_acc", metrics['query_acc'], on_step=True, on_epoch=False, logger=False, prog_bar=True)
        
        return metrics['query_loss']
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ProtoMAML")
        
        # model parameters
        parser.add_argument('--model', default='bert-base-uncased', type=str,
                            help='Which model to train.')
        parser.add_argument('--hidden_size', default=512, type=int,
                            help='Hidden size of the MLP on top of the transformer.')
        parser.add_argument('--output_size', default=512, type=int,
                            help='Output size of the MLP on top of the transformer.')
        parser.add_argument('--inner_updates', default=100, type=int,
                            help='Number of steps taken in the inner loop.')
        parser.add_argument('--eval_updates', default=10, type=int,
                            help='Number of steps taken in the inner loop during evaluation.')

        # Optimizer hyperparameters
        parser.add_argument('--lr', default=5e-5, type=float,
                            help='Learning rate to use.')
        parser.add_argument('--weight_decay', default=0, type=float,
                            help='Weight decay.')
        
        parser.add_argument('--alt_step', action='store_true',
                            help='Whether to use the alternate formulation.')
        parser.add_argument('--gradient_checkpointing', action='store_true',
                            help='Whether to use the gradient checkpointing. Reduces memory usage.')
        
        return parent_parser
        