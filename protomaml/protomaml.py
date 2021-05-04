import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import pytorch_lightning as pl
import torchmetrics.functional as f

from models.BERT import BERT

class ProtoMAML(pl.LightningModule):
    def __init__(self, model='bert-base-uncased', hidden_size=512, output_size=512, inner_updates=100, lr=1e-2, weight_decay=0, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        
        self.model = BERT(model, hidden_size=hidden_size, output_size=output_size)
        self.model.train()
        
        self.protolayer = nn.Linear(output_size, 5)
        
    def feature_forward(self, x):
        return self.model(x)
    
    def forward(self, x):
        x = self.feature_forward(x)
        x = self.protolayer(x)
        return x

    def training_step(self, batch, batch_indx):
        # keep track of the loss and accuracy (for logging)
        support_loss_list = []
        support_acc_list = []
        query_loss_list = []
        query_acc_list = []
        
        # get the optimizer
        opt = self.optimizers()
        
        # loop over sampled tasks
        for task in batch:
            # get the support and query dataloader
            support_loader = task.support_loader
            query_loader = task.query_loader
            
            # create prototype layer
            loader = iter(support_loader)
            for batch_x, batch_y in loader:
                if torch.numel(torch.unique(batch_y)) == task.n_classes:
                    batch_x = [i.to(self.device) for i in batch_x]
                    batch_y = batch_y.to(self.device)
                    pred_y = self.model(batch_x)
                    self.calculate_prototypes(pred_y, batch_y, task.n_classes)
                    break
            del batch_x
            del batch_y
            
            # perform the inner loops
            # clone the model and create a differentiable optimizer
            # higher creates a differentiable inner loop for us
            with higher.innerloop_ctx(self.model, opt.optimizer, copy_initial_weights=False) as (fmodel, diffopt):
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
                    pred_y = self.protolayer(fmodel(batch_x))
                    support_loss = F.cross_entropy(pred_y, batch_y)
                    diffopt.step(support_loss)
                    support_loss_list.append(support_loss.detach())
                    support_acc_list.append(f.accuracy(F.softmax(pred_y.detach(), dim=-1), batch_y.detach()))

                # subsitute the orgininal prototypes back in the grad graph.
                # this might be unnesesairy, but i'm unsure
                self.protolayer.weight = torch.nn.Parameter(2*self.prototypes + (self.protolayer.weight - (2*self.prototypes)).detach())
                self.protolayer.bias = torch.nn.Parameter(-(self.prototypes.norm(dim=-1)**2) + (self.protolayer.bias + (self.prototypes.norm(dim=-1)**2)).detach())

                # abtain the gradient on the query set
                for batch_x, batch_y in query_loader:
                    batch_x = [i.to(self.device) for i in batch_x]
                    batch_y = batch_y.to(self.device)
                    pred_y = self.protolayer(fmodel(batch_x))
                    query_loss = F.cross_entropy(pred_y, batch_y)
                    query_loss_list.append(query_loss.detach())
                    query_acc_list.append(f.accuracy(F.softmax(pred_y.detach(), dim=-1), batch_y.detach()))
                    
                    # calculate the gradients
                    # set create_graph=True for higher order derivatives
                    grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, fmodel.parameters()), retain_graph=True)
                    meta_grads = torch.autograd.grad(query_loss, filter(lambda p: p.requires_grad, self.model.parameters()), retain_graph=True)
                    
                    # save the gradients in the model
                    for param, grad, meta_grad in zip(filter(lambda p: p.requires_grad, self.model.parameters()), grads, meta_grads):
                        if param.grad is not None:
                            param.grad += grad + meta_grad
                        else:
                            param.grad = grad + meta_grad
        
        # update the original model parameters and reset the gradients
        opt.step()
        self.zero_grad(self.model)
        self.zero_grad(self.protolayer)

        # calculate the loss 
        support_loss = torch.stack(support_loss_list).mean()
        query_loss = torch.stack(query_loss_list).mean()
        
        # calculate accuracy
        support_acc = torch.stack(support_acc_list).mean()
        query_acc = torch.stack(query_acc_list).mean()

        # log the loss
        self.log("train_query_loss", query_loss, on_step=False, on_epoch=True)
        self.log("train_support_loss", support_loss, on_step=False, on_epoch=True)
        self.log("train_query_acc", query_acc, on_step=False, on_epoch=True)
        self.log("train_support_acc", support_acc, on_step=False, on_epoch=True)
    
    # not realldy done yet.
    def validation_step(self, batch, batch_indx):
        raise NotImplementedError
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def calculate_prototypes(self, model_output, labels, n_classes):
        prototypes = torch.zeros((n_classes, model_output.shape[-1]), device=self.device)
        for c in range(n_classes):
            indices = torch.nonzero(labels == c).view(-1)
            if indices.numel() != 0:
                prototypes[c] = torch.mean(model_output.index_select(0, indices), dim=0)
        self.prototypes = prototypes
        weight = 2 * prototypes.clone()
        bias = -(prototypes.clone().norm(dim=-1)**2)
        self.protolayer.weight = torch.nn.Parameter(weight.detach())
        self.protolayer.bias = torch.nn.Parameter(bias.detach())
        
    def zero_grad(self, module):
        for param in filter(lambda p: p.requires_grad, module.parameters()):
            param.grad = None
        
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

        # Optimizer hyperparameters
        parser.add_argument('--lr', default=1e-3, type=float,
                            help='Learning rate to use.')
        parser.add_argument('--weight_decay', default=0, type=float,
                            help='Weight decay.')
        
        return parent_parser