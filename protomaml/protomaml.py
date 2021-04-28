import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from BERT import BERT

class ProtoMAML(pl.LightningModule):
    def __init__(self, model='bert-base-uncased', hidden_size=512, output_size=512, inner_updates=100, lr=1e-2, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization=False
        
        self.model = BERT(model, hidden_size=hidden_size, output_size=output_size)
        self.model.train()
        
        self.protolayer = nn.Linear(output_size, 5)
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_indx):
        # keep track of the loss (for logging)
        support_loss_list = []
        query_loss_list = []
        
        # get the optimizer
        opt = self.optimizers()
        
        # loop over sampled tasks
        for task in batch:
            # get the support and query dataloader
            support_loader = task.support_loader
            query_loader = task.query_loader
            
            # create prototype layer
            # how do I ensure I get a prototype for every class?
            batch_x, batch_y = iter(support_loader)
            pred_y = self.model(batch_x)
            self.calculate_prototypes(pred_y, batch_y, task.n_classes)
            
            # perform the inner loops
            # higher creates a differentiable inner loop for us
            with higher.innerloop_ctx(self.model, opt.optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                # ensure that our model is trainable and reset the gradients.
                self.train()
                fmodel.train()
                fmodel.zero_grad()
                
                # perform task adaptation for k inner steps
                for i in self.hparams.inner_updates:
                    batch_x, batch_y = iter(support_loader)
                    pred_y = self.protolayer(fmodel(batch_x))
                    support_loss = F.cross_entropy(pred_y, batch_y)
                    diffopt.step(support_loss)
                    support_loss_list.append(support_loss.detach())

                # subsitute the orgininal prototypes back in the grad graph.
                # this might be unnesesairy, but i'm unsure
                self.protolayer.weight = 2*self.prototypes + (self.protolayer.weight - (2*self.prototypes)).detach()
                self.protolayer.bias = -(self.prototypes.norm(dim=-1)**2) + (self.protolayer.bias + (self.prototypes.norm(dim=-1)**2)).detach()

                # abtain the gradient on the query set
                for batch_x, batch_y in query_loader:
                    pred_y = self.protolayer(fmodel(batch_x))
                    outer_loss = F.cross_entropy(pred_y, batch_y)
                    outer_loss_list.append(outer_loss.detach())
                    
                    # calculate the gradients
                    # set retain_graph=True for higher order derivatives
                    grads = torch.autograd.grad(outer_loss, filter(lambda p: p.requires_grad, fmodel.parameters()), retain_graph=False)
                    meta_grads = torch.autograd.grad(outer_loss, filter(lambda p: p.requires_grad, self.model.parameters()), retain_graph=False)
                    
                    # save the gradients in the model
                    for param, grad, meta_grad in zip(filter(lambda p: p.requires_grad, self.model.parameters()), grads, meta_grads):
                        if param.grad is not None:
                            param.grad += grad + meta_grad
                        else:
                            param.grad = grad + meta_grad
        
        # update the mode parameters and reset the gradients
        opt.step()
        self.model.zero_grad()
        self.protolayer.zero_grad()

        # calculate the loss 
        support_loss = torch.stack(support_loss_list).mean()
        query_loss = torch.stack(outer_loss_list).mean()

        # log the loss
        self.log("train_query_loss", query_loss, on_step=False, on_epoch=True)
        self.log("train_support_loss", support_loss, on_step=False, on_epoch=True)
        return query_loss
    
    # not realldy done yet.
    def validation_step(self, batch, batch_indx):
        raise NotImplementedError
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def calculate_prototypes(self, model_output, labels, n_classes):
        prototypes = torch.zeros((n_classes, model_output.shape[-1]), device=model_output.device)
        for c in range(n_classes):
            indices = torch.nonzero(labels == c).view(-1)
            if indices.numel() != 0:
                prototypes[c] = torch.mean(model_output.index_select(0, indices), dim=0)
        self.prototypes = prototypes
        weight = 2 * prototypes.clone()
        bias = -(prototypes.clone().norm(dim=-1)**2)
        self.protolayer.weight = weight.detach()
        self.protolayer.weight.requires_grad = True
        self.protolayer.bias = bias.detach()
        self.protolayer.bias.requires_grad = True
        
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