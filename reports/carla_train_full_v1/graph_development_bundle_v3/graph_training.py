"""Synthetic-only trainer plus pure frozen epoch/selection/threshold mechanics."""
import copy
import numpy as np
from graph_common import *


def epoch_permutation(count,seed,epoch):
    require(seed in SEEDS and type(epoch) is int and 0<=epoch<100 and type(count) is int and count>0,'Invalid epoch plan')
    return np.random.Generator(np.random.PCG64(seed+epoch)).permutation(count)


class Selection:
    def __init__(self):
        self.tracked_best=self.actual_best=float('inf');self.bad=0;self.selected_epoch=None;self.losses=[]

    def observe(self,epoch,loss):
        require(type(epoch) is int and epoch==len(self.losses) and np.isfinite(loss) and loss>=0,'Invalid epoch/loss')
        self.losses.append(float(loss));changed=False
        if loss<self.actual_best:
            self.actual_best=float(loss);self.selected_epoch=epoch;changed=True
        if loss<self.tracked_best-1e-5: self.tracked_best=float(loss);self.bad=0
        else: self.bad+=1
        return changed,self.bad>=10 and len(self.losses)>=1


def bce_samples(q,y):
    q,y=np.asarray(q,dtype=np.float64),np.asarray(y,dtype=np.float64)
    require(q.shape==y.shape and q.ndim==1 and q.size>0 and np.isfinite(q).all() and
        np.all((q>=0)&(q<=1)) and np.all((y==0)|(y==1)),'BCE domain')
    # Preserve historical development metric clipping/float64 statistics only.
    return -(y*np.log(np.clip(q,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-q,1e-7,1-1e-7)))


def scenario_macro_bce(q,y,scenario_ids,expected_ids):
    ids=np.asarray(scenario_ids);values=bce_samples(q,y)
    require(ids.shape==values.shape and len(expected_ids)==15 and len(set(expected_ids))==15 and
        set(ids)==set(expected_ids),'Validation must cover exactly current 15 scenarios')
    return float(np.mean([values[ids==s].mean() for s in expected_ids]))


def corruption_f1(y_corrupt,predicted):
    y=np.asarray(y_corrupt,dtype=bool);pred=np.asarray(predicted,dtype=bool)
    tp=int(np.count_nonzero(y&pred));fp=int(np.count_nonzero(~y&pred));fn=int(np.count_nonzero(y&~pred))
    denominator=2*tp+fp+fn
    return 2*tp/denominator if denominator else 0.0


def select_threshold(q_normal,target_normal,scenario_ids,expected_ids):
    q=np.asarray(q_normal,dtype=np.float64);y=np.asarray(target_normal);ids=np.asarray(scenario_ids)
    bce_samples(q,y)
    require(ids.shape==q.shape and len(expected_ids)==15 and len(set(expected_ids))==15 and
            set(ids)==set(expected_ids),'Threshold requires current 15-scenario coverage')
    p=1-q;candidates=np.unique(np.concatenate((p,[0.,1.])))
    objectives=[]
    for tau in candidates:
        objectives.append(float(np.mean([corruption_f1(y[ids==s]==0,p[ids==s]>=tau) for s in expected_ids])))
    best=max(objectives);index=max(i for i,v in enumerate(objectives) if v==best)
    return dict(tau=float(candidates[index]),macro_corruption_F1=best,candidates=candidates.tolist(),
        candidate_objectives=objectives,selected_index=index,development_only=True)


def optimizer_for(model):
    import graph_common
    require(graph_common.ACTIVE_PHASE=='synthetic','Real optimizer startup forbidden')
    import torch
    return torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)


def train_synthetic(fixture,seed,*,fixture_epochs=2):
    """Bounded smoke exercise; fixture epoch cap is not a change to real max_epochs=100."""
    from graph_models import SyntheticGraphs,paired_models,torch,F
    require(isinstance(fixture,SyntheticGraphs) and 1<=fixture_epochs<=2,'Synthetic-only bounded trainer')
    train=np.flatnonzero(np.asarray(fixture.roles)=='GRAPH_TRAIN');val=np.flatnonzero(np.asarray(fixture.roles)=='GRAPH_VAL')
    expected=sorted(set(fixture.scenario_ids[val]))
    require(len(train)>0 and len(expected)==15,'Synthetic split coverage')
    models=paired_models(seed,fixture);out={}
    x=torch.from_numpy(fixture.nodes.astype(np.float32));e=torch.from_numpy(fixture.nodes[:,:,1].copy())
    y=torch.from_numpy(fixture.targets)
    for method in METHODS:
        torch.manual_seed(seed);model=models[method];optimizer=optimizer_for(model);selection=Selection();saved=None;orders=[]
        for epoch in range(fixture_epochs):
            order=train[epoch_permutation(len(train),seed,epoch)]
            orders.append([fixture.row_keys[i] for i in order]);model.train()
            for start in range(0,len(order),256):
                batch=order[start:start+256];optimizer.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy(model(x[batch],e[batch]),y[batch],reduction='mean')
                require(torch.isfinite(loss).item(),'Nonfinite synthetic training loss');loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad(): q=model(x[val],e[val]).numpy()
            loss=scenario_macro_bce(q,fixture.targets[val],fixture.scenario_ids[val],expected)
            changed,stop=selection.observe(epoch,loss)
            if changed: saved=copy.deepcopy(model.state_dict())
            if stop: break
        model.load_state_dict(saved);model.eval()
        with torch.no_grad(): q=model(x[val],e[val]).numpy()
        out[method]=dict(selected_epoch=selection.selected_epoch,validation_losses=selection.losses,
            epoch_row_orders=orders,threshold=select_threshold(q,fixture.targets[val],fixture.scenario_ids[val],expected),
            evidence_kind='synthetic_fixture_only',real_model_trained=False)
    return out
