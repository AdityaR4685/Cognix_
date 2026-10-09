import copy
import unittest
import numpy as np
from graph_common import *
from graph_export import *
from current_restore import validate_mapping,restore_current
from synthetic_fixtures import *


class ExportSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.science=science_fixture();cls.agents=fake_agents(cls.science)
        cls.admission=Admission(fixture_partition(),synthetic=True);cls.raw=raw_fixture()

    def pair(self,tick=5,sid='fixture/fit-000',agents=None,parent=None,science=None):
        m=modality_for_recipe(recipe_for_tick(tick))
        raw=causal_window(self.raw['IMU'],tick) if m=='IMU' else self.raw[m]
        return generate_pair(science or self.science,agents or self.agents,self.admission,sid,tick,raw,
            parent or compact_parent(self.science,self.raw,tick))

    def test_exact_pairing(self):
        rows,l=self.pair();self.assertEqual([r['kind'] for r in rows],['clean','pseudo'])
        self.assertEqual([r['target_normal'] for r in rows],[1,0]);self.assertEqual(rows[0]['pair_id'],rows[1]['pair_id'])

    def test_five_recipe_rotation(self):
        self.assertEqual([recipe_for_tick(t) for t in range(1,11)],list(RECIPES[1:]+RECIPES[:1])*2)

    def test_frozen_severities_and_seed(self):
        for t in range(1,6):
            rows,l=self.pair(t);self.assertEqual(l['severity'],SEVERITIES[RECIPES[t%5]])
            self.assertEqual(l['seed'],t);self.assertEqual(l['base_seed'],0)

    def test_causal_imu_bounds(self):
        for t in (1,3,11,12,15):
            np.testing.assert_array_equal(causal_window(self.raw['IMU'],t),self.raw['IMU'][max(0,t-11):t+1])
        _,l=self.pair(3);self.assertEqual((l['window_start_tick'],l['window_end_tick']),(0,3))

    def test_no_effect_removes_both_before_scoring(self):
        science=dict(self.science);parent=compact_parent(science,self.raw,5)
        science['camera_embedding_features']=lambda raw:parent['Camera'].copy()
        class NeverScore:
            def node(self,x): raise AssertionError('No effect guard must precede scoring')
        rows,l=self.pair(5,agents={m:NeverScore() for m in NODE_ORDER},science=science,parent=parent)
        self.assertEqual(rows,[]);self.assertEqual(l['disposition'],'DROP_BOTH')
        self.assertEqual(l['reason'],'compact_modality_no_effect_np_allclose_rtol_1e-12_atol_1e-12')

    def test_mapping_collision_retained(self):
        class Constant:
            def node(self,x): return np.array([.6,.0,.3],dtype=np.float64)
        rows,l=self.pair(agents={m:Constant() for m in NODE_ORDER})
        self.assertEqual(len(rows),2);self.assertTrue(l['canonical_pEA_collision'])

    def test_one_compact_node_and_untouched_bytes(self):
        for t in range(1,6):
            rows,l=self.pair(t);self.assertEqual(l['corrupted_compact_nodes'],1)
            idx=NODE_ORDER.index(l['corrupted_node'])
            for i in range(3):
                if i!=idx:self.assertEqual(rows[0]['nodes'][i].tobytes(),rows[1]['nodes'][i].tobytes())

    def test_node_and_feature_order(self):
        self.assertEqual(NODE_ORDER,('Camera','IMU','Seg'));self.assertEqual(FEATURE_ORDER,('prob_normal','epistemic','aleatoric'))
        rows,_=self.pair();parent=compact_parent(self.science,self.raw,5)
        for i,m in enumerate(NODE_ORDER): np.testing.assert_array_equal(rows[0]['nodes'][i],self.agents[m].node(parent[m]))

    def test_six_edges_no_self(self):
        a=np.asarray(ADJACENCY_LIST);self.assertEqual(int(a.sum()),6);self.assertEqual(int(np.trace(a)),0)

    def test_exact_61_15_admission(self):
        p=fixture_partition();self.assertEqual(sum(self.admission.role(s)=='GRAPH_TRAIN' for s in p['FIT_NORMAL']),61)
        self.assertEqual(sum(self.admission.role(s)=='GRAPH_VAL' for s in p['FIT_NORMAL']),15)

    def test_role_inheritance(self):
        rows,l=self.pair(sid='fixture/fit-075')
        self.assertEqual([r['graph_role'] for r in rows],['GRAPH_VAL','GRAPH_VAL'])

    def test_no_refit_or_recalibration(self):
        for a in self.agents.values():
            with self.assertRaises(PreparationError): a.fit()
            with self.assertRaises(PreparationError): a.fit_calibrator()
        _,l=self.pair();self.assertFalse(l['upstream_refit']);self.assertFalse(l['recalibration'])

    def test_restored_production_mapping_equation(self):
        a=self.agents['Camera'];x=np.full(18,.25,dtype=np.float64)
        scores=np.array([np.exp(-.5*np.sum((x-.02*i)**2)/4) for i in range(5)])
        p=1/(1+np.exp(-(3*scores-1)));entropy=lambda z:-(z*np.log(z)+(1-z)*np.log(1-z))
        expected=np.array([p.mean(),max(0,float(entropy(p.mean())-entropy(p).mean())),entropy(p).mean()])
        np.testing.assert_allclose(a.node(x),expected,rtol=0,atol=1e-15)
        self.assertFalse(a.ensemble._members[0]._mean.flags.writeable)

    def test_deterministic_pair_id_row_key(self):
        a,la=self.pair();b,lb=self.pair()
        self.assertEqual(la,lb);self.assertEqual([r['row_key'] for r in a],[r['row_key'] for r in b])
        self.assertNotEqual(pair_identity('fixture/fit-000',5),pair_identity('fixture/fit-000',6))

    def test_deterministic_scientific_content_hash(self):
        a,la=self.pair();b,lb=self.pair()
        self.assertEqual(scientific_hash(a,[la],{'fixture':True}),scientific_hash(b,[lb],{'fixture':True}))
        b[0]['nodes'][0,0]+=1e-4
        self.assertNotEqual(scientific_hash(a,[la],{'fixture':True}),scientific_hash(b,[lb],{'fixture':True}))

    def test_complete_pair_validator(self):
        rows=[];ledgers=[]
        for t in range(1,6):
            pair,l=self.pair(t);rows.extend(pair);ledgers.append(l)
        validate_pairs(rows,ledgers,self.admission,'fixture/fit-000',complete_ticks=range(1,6))


class ExportAdversarial(ExportSynthetic):
    # Only these tests are adversarial; inherited mechanics are removed by the runner.
    def test_CAL_excluded(self):
        with self.assertRaises(PreparationError): self.admission.role('fixture/cal-000')

    def test_foreign_and_TEST_parent_excluded(self):
        for sid,split in (('fixture/missing','train'),('fixture/fit-000','test')):
            with self.assertRaises(PreparationError): self.admission.role(sid,split)

    def test_wrong_role_count_rejected(self):
        p=fixture_partition();p['GRAPH_TRAIN']=p['GRAPH_TRAIN'][:-1]
        with self.assertRaises(PreparationError): Admission(p,synthetic=True)

    def test_real_current_restoration_blocked(self):
        with self.assertRaises(PreparationError): restore_current(self.science)

    def test_nonfinite_domain_and_dtype_rejected(self):
        for node in (np.array([np.nan,0,0]),np.array([1.1,0,0]),np.array([.5,-.1,0]),np.array([.5,0,0],dtype=np.float32)):
            with self.assertRaises(PreparationError): check_node(node)
        p=compact_parent(self.science,self.raw,5);p['Camera'][0]=np.inf
        with self.assertRaises(PreparationError): self.pair(parent=p)

    def test_invalid_tick_rejected(self):
        for tick in (0,3000,True,1.5):
            with self.assertRaises(PreparationError): recipe_for_tick(tick)

    def test_duplicate_missing_candidate_rejected(self):
        rows,l=self.pair()
        with self.assertRaises(PreparationError): validate_pairs(rows,[l,l],self.admission,'fixture/fit-000',complete_ticks=[5,5])
        with self.assertRaises(PreparationError): validate_pairs(rows,[l],self.admission,'fixture/fit-000',complete_ticks=[5,6])

    def test_historical_N20_paths_rejected(self):
        for p in ('cache_train20_v1/x.npz','carla_train_expand20/a','reports/carla_gat_paired_execution_v1/a',
                  'reports/carla_final_evaluation_v1/a','reports/carla_graph_fit_export_v1/a'):
            with self.assertRaises(PreparationError): reject_n20_path(p)

    def test_real_artifact_write_blocked(self):
        with self.assertRaises(PreparationError): write_scenario(DATA,'fixture/fit-000',[],[],self.admission,{})

    def test_guard_TEST_network_source_and_writes(self):
        guard,c=make_guard('prepare')
        cases=[('open',(r'E:\carlanomaly-base-test.tar.gz','rb',os.O_RDONLY)),('socket.connect',(None,)),
            ('open',(SOURCE['path'],'rb',os.O_RDONLY)),('open',(str(HISTORICAL_EXPORTER),'wb',os.O_WRONLY)),
            ('open',(str(RUNS/'x'),'wb',os.O_WRONLY)),('open',(str(REPORT/'gate2_train_health_v3/FINAL.json'),'wb',os.O_WRONLY))]
        for event,args in cases:
            with self.assertRaises(PreparationError): guard(event,args)
        self.assertEqual(c['TEST_requests'],0);self.assertEqual(c['network_requests'],0)

    def test_historical_semantic_reads_blocked(self):
        guard,c=make_guard('prepare')
        with self.assertRaises(PreparationError): guard('open',(str(REPO/'reports/carla_gat_paired_execution_v1/results.json'),'rb',os.O_RDONLY))

    def test_ambiguous_mapping_rejected(self):
        bad=dict(status='VALID',require_valid_succeeded=True,outputs_complete=True,
            calibrator_runtime_class='EnsemblePredictiveCalibrator (unchanged production agent)',
            calibrator_parameters=[3.,-1.,.5,.2],optimization_info=dict(finite_optimum=True))
        with self.assertRaises(PreparationError): validate_mapping(bad,np.array([4.,-1.,.5,.2]))

    def test_real_replay_sink_blocked(self):
        from graph_replay import FITSink
        with self.assertRaises(PreparationError): FITSink(None,None,None,None,None,None,None)
