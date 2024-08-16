"""
Created on Thu Aug 28 2023
@author: Kichang Lee
@contact: kichang.lee@yonsei.ac.kr
references
1. code: https://github.com/mit-han-lab/dlg
2. paper: https://arxiv.org/abs/1906.08935
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import copy

from .base import *
import config as cfg
from rgap_utils import *

__all__ = ['RGAP']

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.ones(target.size(0), num_classes, device=target.device) * (-1)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


class RGAP(Attacker):
    def __init__(self, args):
        super().__init__(args)
        self.optimizer = None
        self.iteration = 300
        self.history   = []
        self.criterion = cross_entropy_for_onehot
        pass
    
    def attack(self, old_model, new_model):
        old_model.to(self.device)
        new_model.to(self.device)

        input_size = list(cfg.IMGSIZE[self.args.dataset])
        
        # consider only weights rather than bias
        dy_dx = []
        for new_key, old_key in zip(new_model.state_dict().keys(), old_model.state_dict().keys()):
            if 'weight' in new_key and 'weight' in old_key:
                dy_dx.append(old_model.state_dict()[old_key] - new_model.state_dict()[new_key])
        if self.args.optimizer == 'SGD':                # the weight difference it self is not identical to the gradient!!
            dy_dx = [_/self.args.lr for _ in dy_dx]
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

        # reconstruction procedure
        # Modules: list of layer parameters
        # Nonact Layers: list of "Non-activation" layer parameters
        modules = []
        nonact_layers = []
        for _, module in old_model.named_modules():
            if isinstance(module, nn.Module) and not isinstance(module, nn.Sequential) and "torch.nn.modules" in module.__module__: 
                if not is_act(module):
                    nonact_layers.append(module)
                modules.append(module)
        
        # params: list of shapes of parameter matrices
        params = []
        # biases: list of biases of parameter matrices
        biases = []
        for key, param in old_model.state_dict().items():
            if "weight" in key:
                params.append(list(param.size()))
            if "bias" in key:
                biases.append(list(param.size()))

        # Shapes: list of shapes of intermediate inputs
        shapes = []
        input_size = list(cfg.IMGSIZE[self.args.dataset])
        test_input = torch.rand(input_size).to(self.args.device)
        print(test_input.size())
        for layer in nonact_layers:
            if isinstance(layer, nn.Linear):
                test_input = test_input.flatten(1)
            input_size = list(test_input.size())
            shapes.append(input_size)
            test_input = layer(test_input)

        # Start from the last layer, which is the output
        original_dy_dx.reverse()
        modules.reverse()
        nonact_layers.reverse()
        params.reverse()
        biases.reverse()
        shapes.reverse()

        print("original_dy_dx: ", end="")
        for dy_dx in original_dy_dx:
            print(dy_dx.size(), end=" ")
        print("")
        print("modules: ", modules)
        print("layers: ", nonact_layers)
        print("params: ", params)
        print("shapes: ", shapes)

        k = None
        da = None
        out = None
        o_idx = 0   # original index
        padding = 0
        
        last_weight = []

        for layer in modules:
            print(layer)

        # perform r-gap for each layer
        for m_idx in range(len(modules)):
            print("====================================")
            print(f"{m_idx+1}. Peeling {modules[m_idx]} - ", end="")
            # No parameters exist: activation layer
            if k is not None and is_act(modules[m_idx]):
                print("Activation")
                # derive activation function
                if isinstance(modules[m_idx], nn.LeakyReLU):
                    da = derive_leakyrelu(x_, slope=modules[m_idx].negative_slope)
                elif isinstance(modules[m_idx], nn.Identity):
                    da = derive_identity(x_)
                elif isinstance(modules[m_idx], nn.Sigmoid):
                    da = derive_sigmoid(x_)
                elif isinstance(modules[m_idx], nn.GELU):
                    da = derive_gelu(x_)
                else:
                    ValueError(f'Please implement the derivative function of {modules[m_idx]}')

                # back out neuron output
                if isinstance(modules[m_idx], nn.LeakyReLU):
                    out = inverse_leakyrelu(x_, slope=modules[m_idx].negative_slope)
                elif isinstance(modules[m_idx], nn.Identity):
                    out = inverse_identity(x_)
                elif isinstance(modules[m_idx], nn.Sigmoid):
                    out = inverse_sigmoid(x_)
                elif isinstance(modules[m_idx], nn.GELU):
                    out = inverse_gelu(x_)
                else:
                    ValueError(f'Please implement the inverse function of {modules[m_idx]}')

                continue
            
            # Parameters exist: normal layer
            else:
                print("Layer")
                gradient_i = original_dy_dx[o_idx].to(self.args.device)
                weight_i = list(modules[m_idx].parameters())[0].to(self.args.device)
                bias_i = list(modules[m_idx].parameters())[1].to(self.args.device)

                """print(f"- init g:      {gradient_i.shape}")
                print(f"- init w:      {weight_i.shape}")"""

                # Layer = dth (Last)
                # Assume the model does not end with activation function
                if k is None:
                    udldu = torch.dot(gradient_i.reshape(-1), weight_i.reshape(-1))
                    u = inverse_udldu(udldu)

                    
                    # extract ground truth label
                    y = torch.argmin(torch.sum(gradient_i, dim=-1), dim=-1).long().reshape(1,)
                    print(f"Evaluated ground truth label is {y}")
                    y = label_to_onehot(y).float().to(self.args.device)
                    # y = torch.tensor([-1 if n == 0 else n for n in y], dtype=np.float32).reshape(-1, 1)
                    # y = y.mean() if y.mean() != 0 else 0.1

                    print(f'pred_: {u/y}, udldu: {udldu:.1e}, udldu_:{-u/(1+torch.exp(u)):.1e}')
                    k = -y/(1+torch.exp(u))
                    k = k.reshape(-1, 1).float()

                # Layer: 1st to (d-1)th 
                else:
                    for d in range(1, m_idx+1):
                        if not is_act(modules[m_idx-d]):
                            if hasattr(modules[m_idx-d], 'padding'):
                                padding = modules[m_idx-d].padding[0]
                            else:
                                padding = 0
                            break

                    # For a mini-batch setting, reconstruct the combination
                    in_shape = torch.tensor(shapes[o_idx-1]).to(self.args.device)
                    """print(f"- in_shape:    {in_shape}")"""
                    in_shape[0] = 1
                    # peel off padded entries
                    x_mask = peeling(in_shape=in_shape, padding=padding).bool()
                    """print(f"- new in_shape:{in_shape}")
                    print(f"- last_weight: {last_weight.shape}")
                    print(f"- x_mask:      {x_mask.shape}")
                    print(f"- k:           {k.shape}")
                    print(f"- derive act:  {da.shape}")
                    print(f"- last pad:    {padding}")"""
                    k = torch.mul(torch.matmul(last_weight.t(), k)[x_mask], da.t())

                if isinstance(modules[m_idx], nn.Conv2d):
                    x_, last_weight = r_gap(out=out, k=k, x_shape=shapes[o_idx], module=modules[m_idx], g=gradient_i, weight=weight_i)
                else:
                    # In consideration of computational efficiency, for FCN only takes gradient constraints into account.
                    x_, last_weight = fcn_reconstruction(k=k, gradient=gradient_i), weight_i
                """print(f"- x_:          {x_.shape}")
                print(f"- recalc k:    {k.shape}")
                print(f"- recalc lw:   {last_weight.shape}")"""
                
                o_idx += 1
                self.history.append(x_)

        """
        # Optimization-based Method
        optimizer = optim.LBFGS([dummy_data, dummy_label])
        
        
        for iters in range(self.iteration):
            def closure():
                optimizer.zero_grad()
                
                dummy_pred = old_model(dummy_data)
                dummy_onehot_label = F.softmax(dummy_label, dim=-1)
                dummy_loss = self.criterion(dummy_pred, dummy_onehot_label)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, old_model.parameters(), create_graph=True)\
                    
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                return grad_diff
            optimizer.step(closure)
            if iters % 10 == 0: 
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())
                self.history.append(transforms.ToPILImage()(dummy_data[0].cpu()))
        """
 


    