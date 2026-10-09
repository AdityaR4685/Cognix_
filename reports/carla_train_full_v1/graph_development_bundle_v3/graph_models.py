"""Frozen architectures exercised on bounded synthetic fixtures only in this preparation."""
import ast
import copy
from types import SimpleNamespace
from typing import Optional
import numpy as np
import graph_common
from graph_common import *
require(graph_common.ACTIVE_PHASE=='synthetic','Model implementation is preparation-synthetic only')
import torch
from torch import nn
from torch.nn import functional as F


def exact_generic_definitions():
    path=REPO/'cognix/graph/epistemic_gat.py'
    frozen=read_json(BUNDLE/'current_experiment_bindings.json')['scientific_source_hashes']
    require(hash_file(path)==frozen[path.relative_to(REPO).as_posix()],'Generic GAT implementation changed')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    layer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpistemicGATLayerPT')
    full=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EpistemicGAT')
    weights=next(n for n in full.body if isinstance(n,ast.FunctionDef) and n.name=='compute_epistemic_weights')
    ns=dict(torch=torch,nn=nn,F=F,np=np,Optional=Optional,__name__=__name__)
    # Execute these exact unedited AST bodies; broad cognix startup is excluded.
    exec(compile(ast.Module(body=[layer,weights],type_ignores=[]),str(path),'exec'),ns)
    return ns['EpistemicGATLayerPT'],ns['compute_epistemic_weights']


GenericLayer,compute_epistemic_weights=exact_generic_definitions()


def cpu_determinism():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


class SyntheticGraphs:
    def __init__(self,nodes,targets,scenario_ids,row_keys,roles):
        require(graph_common.ACTIVE_PHASE=='synthetic' and isinstance(nodes,np.ndarray) and nodes.dtype==np.float64 and
            nodes.ndim==3 and nodes.shape[1:]==(3,3) and 1<=len(nodes)<=512,'Bounded synthetic graph input required')
        require(np.isfinite(nodes).all() and np.all((nodes[:,:,0]>=0)&(nodes[:,:,0]<=1)) and
            np.all(nodes[:,:,1:]>=0) and np.all(nodes[:,:,1:]<=np.log(2)+1e-12),'Synthetic node domain')
        n=len(nodes)
        require(len(targets)==len(scenario_ids)==len(row_keys)==len(roles)==n and len(set(row_keys))==n and
            all(s.startswith('fixture/') for s in scenario_ids) and all(k.startswith('fixture:') for k in row_keys) and
            all(r in ('GRAPH_TRAIN','GRAPH_VAL') for r in roles) and set(targets)<=set((0,1)),
            'Real/foreign data cannot enter preparation model fixtures')
        self.nodes,self.targets=nodes.copy(),np.asarray(targets,dtype=np.float32)
        self.scenario_ids,self.row_keys,self.roles=np.asarray(scenario_ids),tuple(row_keys),tuple(roles)


class NoGraph(nn.Module):
    def __init__(self):
        super().__init__();self.first=nn.Linear(3,12,bias=False);self.last=nn.Linear(12,1,bias=False)
        self.dropout=nn.Dropout(.1)
        nn.init.xavier_uniform_(self.first.weight);nn.init.xavier_uniform_(self.last.weight)

    def forward(self,nodes,original_e=None):
        logits=self.last(self.dropout(F.elu(self.first(nodes))))
        return torch.sigmoid(logits.squeeze(-1)).mean(-1)


class BatchedGAT(nn.Module):
    def __init__(self,use_prior):
        super().__init__();self.layers=nn.ModuleList([GenericLayer(3,8,.1),GenericLayer(8,1,.1)])
        self.use_prior=use_prior;self.register_buffer('adjacency',torch.tensor(ADJACENCY_LIST,dtype=torch.float32))

    def prior(self,original_e):
        require(original_e.dtype==torch.float64 and original_e.ndim==2 and original_e.shape[1]==3 and
            torch.isfinite(original_e).all().item() and (original_e>=0).all().item(),'Original canonical E required')
        # Reciprocal in source float64, then float32 exactly as generic Python float -> np.float32.
        w=(1.0/(1.0+original_e)).to(torch.float32) if self.use_prior else torch.ones_like(original_e,dtype=torch.float32)
        return w[:,None,:].expand(-1,3,-1)

    def layer_forward(self,layer,h,prior,final):
        wh=layer.W(h);n=3
        pair=torch.cat((wh[:,:,None,:].expand(-1,-1,n,-1),wh[:,None,:,:].expand(-1,n,-1,-1)),dim=-1)
        e=layer.leaky_relu(layer.a(pair).squeeze(-1))+torch.log(prior)
        a=self.adjacency
        mask=torch.where(a>0,torch.zeros_like(a),torch.full_like(a,-1e9))
        alpha=F.softmax(e+mask,dim=-1)*a
        alpha=alpha/alpha.sum(-1,keepdim=True).clamp(min=1e-9)
        alpha=layer.dropout(alpha)
        aggregate=alpha@wh
        return aggregate if final else F.elu(aggregate),alpha

    def forward(self,nodes,original_e):
        prior=self.prior(original_e)  # One original E reused unchanged for both layers.
        h=nodes
        for i,layer in enumerate(self.layers): h,_=self.layer_forward(layer,h,prior,i==1)
        return torch.sigmoid(h.squeeze(-1)).mean(-1)


def seed_plan(seeds=SEEDS):
    require(tuple(seeds)==SEEDS,'Exactly five paired seeds; no best-seed selection/replacement')
    return tuple((seed,method) for seed in SEEDS for method in METHODS)


def paired_models(seed,fixture):
    require(graph_common.ACTIVE_PHASE=='synthetic' and isinstance(fixture,SyntheticGraphs) and seed in SEEDS,
        'Synthetic fixture/paired allowed seed required')
    cpu_determinism();torch.manual_seed(seed);no_graph=NoGraph()
    torch.manual_seed(seed);standard=BatchedGAT(False)
    epistemic=copy.deepcopy(standard);epistemic.use_prior=True
    models=dict(NoGraph=no_graph,StandardGAT=standard,EpistemicGAT=epistemic)
    require([sum(p.numel() for p in models[m].parameters()) for m in METHODS]==[48,50,50],'Parameter counts differ')
    return models
