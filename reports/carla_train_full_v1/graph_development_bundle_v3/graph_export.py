"""Current FIT-only graph pair mechanics; no raw reader or historical exporter import."""
import io
import numpy as np
import graph_common
from graph_common import *


def recipe_for_tick(tick):
    require(type(tick) is int and 1<=tick<=2999,'Parent tick outside 1..2999')
    return RECIPES[tick%5]


def modality_for_recipe(recipe):
    require(recipe in RECIPES,'Foreign recipe')
    return 'Camera' if recipe.startswith('camera') else 'Seg' if recipe.startswith('seg') else 'IMU'


def causal_window(imu,tick):
    recipe_for_tick(tick)
    require(imu.ndim==2 and imu.shape[1]==3 and len(imu)>tick,'Invalid causal IMU table')
    return imu[max(0,tick-11):tick+1].copy()


class Admission:
    def __init__(self,partition,*,synthetic=False):
        require(len(partition['GRAPH_TRAIN'])==61 and len(partition['GRAPH_VAL'])==15 and
            len(set(partition['GRAPH_TRAIN']+partition['GRAPH_VAL']))==76 and
            set(partition['GRAPH_TRAIN']+partition['GRAPH_VAL'])==set(partition['FIT_NORMAL']) and
            len(partition['CAL_NORMAL_excluded'])==25 and
            not set(partition['FIT_NORMAL'])&set(partition['CAL_NORMAL_excluded']),'61/15 FIT-only admission failed')
        if synthetic:
            require(graph_common.ACTIVE_PHASE=='synthetic' and
                all(s.startswith('fixture/') for s in partition['FIT_NORMAL']+partition['CAL_NORMAL_excluded']),
                'Synthetic fixture cannot admit a real scenario')
        else:
            require(graph_common.ACTIVE_PHASE=='graph-data','Real admission requires future authorized process')
            frozen=read_json(BUNDLE/'current_partition_audit.json')
            require(partition==frozen,'Current partition must equal sealed audited membership')
        self.partition,self.synthetic=partition,synthetic
        self.roles={s:r for r in ('GRAPH_TRAIN','GRAPH_VAL') for s in partition[r]}

    def role(self,sid,source_split='train'):
        require(source_split=='train' and sid in self.roles and sid not in self.partition['CAL_NORMAL_excluded'],
                'Foreign/TEST/CAL pseudo parent forbidden')
        reject_n20_path(sid)
        return self.roles[sid]


def check_node(node):
    require(isinstance(node,np.ndarray) and node.shape==(3,) and node.dtype==np.float64 and
            np.isfinite(node).all() and 0<=node[0]<=1 and np.all(node[1:]>=0) and
            np.all(node[1:]<=np.log(2)+1e-12),'Invalid canonical p/E/A domain/order')


def pair_identity(sid,tick):
    return digest(canonical(dict(schema='Experiment-2B-graph-pair-v1',scenario=sid,tick=tick,
        partition_sha256=GRAPH_PARTITION_SHA,Gate2_runtime_seal=SEALS['gate2_train_health_v3'],
        protocol_scientific_sha256=PROTOCOL_SHA)))


def generate_pair(science,agents,admission,sid,tick,raw_payload,parent,*,source_split='train'):
    role=admission.role(sid,source_split)
    recipe=recipe_for_tick(tick);modality=modality_for_recipe(recipe)
    require(set(parent)==set(NODE_ORDER) and set(agents)==set(NODE_ORDER),'Node order/schema failed')
    for m in NODE_ORDER:
        require(parent[m].dtype==np.float64 and parent[m].shape==(DIMS[m],) and
                np.isfinite(parent[m]).all(),'Compact feature finite/dimension guard')
    raw=np.asarray(raw_payload).copy();preprocessing=None
    if modality=='Seg': raw,preprocessing=science['sanitize'](raw)
    fn=science[dict(Camera='camera_embedding_features',IMU='imu_window_features',
                    Seg='segmentation_histogram_features')[modality]]
    sample=science['generate_pseudo_anomaly']({modality:raw},recipe_id=recipe,
        source_split=science['CarlAnomalySplit'].TRAIN,source_scenario=sid,source_tick=tick,
        seed=tick,severity=SEVERITIES[recipe])
    corrupt=np.asarray(fn(np.asarray(sample.data)))
    require(corrupt.dtype==np.float64 and corrupt.shape==(DIMS[modality],) and
            np.isfinite(corrupt).all(),'Corrupted feature finite/dimension guard')
    provenance=sample.provenance()
    require(provenance['recipe_id']==recipe and provenance['seed']==tick and
        provenance['severity']==SEVERITIES[recipe] and provenance['source_split']=='train' and
        provenance['source_scenario']==sid and provenance['source_tick']==tick and
        provenance['synthetic_corruption'] is True,'Recipe provenance disagreement')
    pair_id=pair_identity(sid,tick)
    ledger=dict(pair_id=pair_id,parent_scenario=sid,parent_tick=tick,graph_role=role,recipe_id=recipe,
        severity=SEVERITIES[recipe],seed=tick,base_seed=0,corrupted_node=modality,provenance=provenance,
        window_start_tick=max(0,tick-11) if modality=='IMU' else tick,window_end_tick=tick,
        segmentation_preprocessing=preprocessing,compact_parent_sha256=digest(parent[modality].tobytes()),
        compact_pseudo_sha256=digest(corrupt.tobytes()),upstream_refit=False,recalibration=False)
    if np.allclose(corrupt,parent[modality],rtol=1e-12,atol=1e-12):
        ledger.update(disposition='DROP_BOTH',reason='compact_modality_no_effect_np_allclose_rtol_1e-12_atol_1e-12')
        return [],ledger
    clean=np.stack([agents[m].node(parent[m]) for m in NODE_ORDER])
    for node in clean: check_node(node)
    pseudo=clean.copy();idx=NODE_ORDER.index(modality)
    pseudo[idx]=agents[modality].node(corrupt);check_node(pseudo[idx])
    for i in range(3):
        if i!=idx: require(pseudo[i].tobytes()==clean[i].tobytes(),'Untouched node bytes changed')
    collision=pseudo[idx].tobytes()==clean[idx].tobytes()
    ledger.update(disposition='RETAIN_PAIR',canonical_pEA_collision=collision,
                  no_effect_selection_uses_pEA=False,corrupted_compact_nodes=1)
    rows=[]
    for kind,target,nodes in (('clean',1,clean),('pseudo',0,pseudo)):
        rows.append(dict(row_key=pair_id+':'+kind,pair_id=pair_id,kind=kind,target_normal=target,
            scenario_id=sid,parent_tick=tick,graph_role=role,nodes=nodes))
    return rows,ledger


def scientific_hash(rows,ledgers,binding):
    h=hashlib.sha256(canonical(dict(schema='graph-scientific-content-v1',binding=binding,
        node_order=NODE_ORDER,feature_order=FEATURE_ORDER,adjacency=ADJACENCY_LIST,ledgers=ledgers)))
    for row in rows:
        node=row['nodes'];require(node.shape==(3,3) and node.dtype==np.float64,'Graph node dtype/shape')
        for v in node: check_node(v)
        metadata={k:v for k,v in row.items() if k!='nodes'}
        require(row['row_key']==row['pair_id']+':'+row['kind'] and row['target_normal']==
                (1 if row['kind']=='clean' else 0) and row['kind'] in ('clean','pseudo'),'Row identity/target guard')
        h.update(canonical(metadata));h.update(canonical(dict(dtype='<f8',shape=[3,3])))
        h.update(node.astype('<f8',copy=False).tobytes(order='C'))
    return h.hexdigest()


def validate_pairs(rows,ledgers,admission,sid,*,complete_ticks):
    role=admission.role(sid)
    ticks=[l['parent_tick'] for l in ledgers]
    require(len(ticks)==len(set(ticks)) and ticks==list(complete_ticks),'Duplicate/missing/reordered candidates')
    require(len({r['row_key'] for r in rows})==len(rows),'Duplicate graph row keys')
    position=0
    for ledger in ledgers:
        t=ledger['parent_tick'];recipe=recipe_for_tick(t);m=modality_for_recipe(recipe)
        require(ledger['pair_id']==pair_identity(sid,t) and ledger['graph_role']==role and
            ledger['recipe_id']==recipe and ledger['seed']==t and ledger['severity']==SEVERITIES[recipe] and
            ledger['corrupted_node']==m,'Candidate ledger binding failed')
        if ledger['disposition']=='DROP_BOTH':
            require(ledger['reason']=='compact_modality_no_effect_np_allclose_rtol_1e-12_atol_1e-12','Foreign drop reason')
            continue
        require(ledger['disposition']=='RETAIN_PAIR','Foreign disposition')
        a,b=rows[position:position+2];position+=2
        require(a['kind']=='clean' and b['kind']=='pseudo' and a['target_normal']==1 and b['target_normal']==0 and
            a['pair_id']==b['pair_id']==ledger['pair_id'] and a['graph_role']==b['graph_role']==role and
            a['scenario_id']==b['scenario_id']==sid,'Exact pairing/role inheritance failed')
        for i in range(3):
            if i!=NODE_ORDER.index(m): require(a['nodes'][i].tobytes()==b['nodes'][i].tobytes(),'Untouched bytes mismatch')
    require(position==len(rows),'Extra unpaired graph row')


def write_scenario(root,sid,rows,ledgers,admission,binding):
    require(graph_common.ACTIVE_PHASE=='graph-data','Real graph artifact writing forbidden in preparation')
    validate_pairs(rows,ledgers,admission,sid,complete_ticks=range(1,3000))
    root.mkdir()
    buffer=io.BytesIO()
    np.savez_compressed(buffer,nodes=np.asarray([r['nodes'] for r in rows],dtype=np.float64).reshape(-1,3,3),
        target_normal=np.asarray([r['target_normal'] for r in rows],dtype=np.float32),
        row_key=np.asarray([r['row_key'] for r in rows],dtype='U71'),
        adjacency=np.asarray(ADJACENCY_LIST,dtype=np.float32))
    write_new(root/'graphs.npz',buffer.getvalue())
    write_json(root/'rows.json',[{k:v for k,v in r.items() if k!='nodes'} for r in rows])
    write_json(root/'candidate_ledger.json',ledgers)
    summary=dict(scenario_id=sid,graph_role=admission.role(sid),candidates=2999,rows=len(rows),pairs=len(rows)//2,
        dropped_pairs=sum(l['disposition']=='DROP_BOTH' for l in ledgers),
        collisions=sum(l.get('canonical_pEA_collision',False) for l in ledgers),binding=binding,
        scientific_content_sha256=scientific_hash(rows,ledgers,binding))
    write_json(root/'summary.json',summary)
    summary['unit_seal']=seal_tree(root,'artifact_inventory.json')
    return summary
