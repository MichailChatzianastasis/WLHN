import networkx as nx
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_geometric.nn import GINConv
from torch_scatter import scatter_add, scatter_mean
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder

from hypernn import MobiusMLR, MobiusLinear

MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

class WLHN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, tau, classifier, n_classes, dropout):
        super(WLHN, self).__init__()
        self.n_layers = n_layers
        self.classifier = classifier
        
        self.scaling = torch.tanh(torch.tensor(tau / 2))
        self.atom_encoder = AtomEncoder(hidden_dim)

        self.fc0 = nn.Linear(input_dim, hidden_dim)

        self.p = torch.zeros(hidden_dim, requires_grad=False)
        self.p[-1] = 1 

        lst = list()
        lst.append(GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))

        for i in range(n_layers-1):
            lst.append(GINConv(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim),
                           nn.Linear(hidden_dim, hidden_dim), nn.ReLU())))

        self.conv = nn.ModuleList(lst)

        if classifier == 'hyperbolic_mlr':
            self.fc1 = MobiusLinear(hidden_dim, 128)
            self.fc2 = MobiusLinear(128, 64)
            self.fc3 = MobiusMLR(64, n_classes)
        elif classifier == 'logmap':
            self.fc1 = nn.Linear(hidden_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, n_classes)
        else:
            raise Exception("Choose a valid classifier classifier!")

        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()


    # Reflection (circle inversion of x through orthogonal circle centered at a)
    def isometric_transform(self, x, a):
        r2 = torch.sum(a ** 2, dim=-1, keepdim=True) - 1.
        u = x - a
        return r2 / torch.sum(u ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM) * u + a


    # center of inversion circle
    def reflection_center(self, mu):
        return mu / torch.sum(mu ** 2, dim=-1, keepdim=True).clamp_min(MIN_NORM)


    # Map x under the isometry (inversion) taking mu to origin
    def reflect_at_zero(self, x, mu):
        a = self.reflection_center(mu)
        return self.isometric_transform(x, a)


    # Image of x under reflection that takes p (normalized) to q (normalized) and 0 to 0
    def reflect_through_zero(self, p, q, x):
        p_ = p / torch.norm(p, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        q_ = q / torch.norm(q, dim=-1, keepdim=True).clamp_min(MIN_NORM)
        r = q_ - p_
        # Magnitude of x in direction of r
        m = torch.sum(r * x, dim=-1, keepdim=True) / torch.sum(r * r, dim=-1, keepdim=True)
        return x - 2 * r * m


    def project(self, x):
        """Project points to Poincare ball with curvature c.
        Args:
            x: torch.Tensor of size B x d with hyperbolic points
        Returns:
            torch.Tensor with projected hyperbolic points.
        """
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        eps = BALL_EPS[x.dtype]
        maxnorm = (1 - eps)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    
    def expmap0(self, u):
        """Exponential map taken at the origin of the Poincare ball with curvature c.
        Args:
            u: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
        Returns:
            torch.Tensor with tangent points shape (B, d)
        """
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        gamma_1 = torch.tanh(u_norm) * u / u_norm
        return self.project(gamma_1)


    def logmap0(self, y):
        """Logarithmic map taken at the origin of the Poincare ball with curvature c.
        Args:
            y: torch.Tensor of size B x d with tangent points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures
        Returns:
            torch.Tensor with hyperbolic points.
        """
        y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        return y / y_norm / 1. * torch.atanh(y_norm.clamp(-1 + 1e-15, 1 - 1e-15))


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        #x = self.atom_encoder(x)
        x = self.relu(self.fc0(x))
        xs = [x]
        z = [torch.zeros(1, x.size(1), device=x.device, requires_grad=False)]
        inv = [torch.zeros(x.size(0), dtype=torch.long, device=x.device, requires_grad=False)]
        with torch.no_grad():
            unique_all, inv_all = torch.unique(x, sorted=False, return_inverse=True, dim=0)
        unique_norm_all = self.project(unique_all)
        z.append(self.scaling*unique_norm_all)
        inv.append(inv_all)
        for i in range(self.n_layers):
            x = self.conv[i](x, edge_index)
            diag = torch.zeros(x.size(1), x.size(1)-1, device=x.device, requires_grad=False)
            diag.fill_diagonal_(1)
            diag = torch.cat([diag, torch.zeros(x.size(1), 1, device=x.device)], dim=1)
            x = torch.mm(x, diag)
            xs.append(x)
            with torch.no_grad():
                unique_all, inv_all, count_all = torch.unique(torch.cat(xs, dim=1), sorted=False, return_inverse=True, return_counts=True, dim=0)
            
            unique_all = unique_all[:,-x.size(1):]
            unique_all_norm = self.project(unique_all)
            z_children = self.scaling*unique_all_norm
            t = torch.zeros(unique_all.size(0), dtype=torch.long, device=x.device)
            t.scatter_add_(0, inv_all, inv[i+1])
            t = torch.div(t, count_all).long()
            z_current = torch.gather(z[i+1], 0, t.unsqueeze(1).repeat(1, z[i+1].size(1)))
            t = torch.zeros(unique_all.size(0), dtype=torch.long, device=x.device)
            t.scatter_add_(0, inv_all, inv[i])
            t = torch.div(t, count_all).long()
            z_parent = torch.gather(z[i], 0, t.unsqueeze(1).repeat(1, z[i].size(1)))
            z_parent = self.reflect_at_zero(z_parent, z_current)
            z_children = self.reflect_through_zero(z_parent, self.p.to(x.device), z_children)
            z_all = self.reflect_at_zero(z_children, z_current)
            inv.append(inv_all)
            z.append(z_all)
        
        x = self.logmap0(z[-1])
        x = torch.index_select(x, 0, inv[-1])
        out = scatter_add(x, data.batch, dim=0)
        
        if self.classifier == 'hyperbolic_mlr':
            out = self.expmap0(out)            
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            out = self.dropout(out)
            out = self.fc3(out)
        else:
            out = self.relu(self.fc1(out))
            out = self.relu(self.fc2(out))
            out = self.fc3(out)
        return F.log_softmax(out, dim=1)
