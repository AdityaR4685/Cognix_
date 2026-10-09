import unittest
from types import SimpleNamespace
import numpy as np
from graph_common import *
from graph_models import *
from graph_training import *
from synthetic_fixtures import model_fixture


class ModelSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.fixture=model_fixture();cls.models=paired_models(101,cls.fixture)

    def test_NoGraph_48_parameters(self): self.assertEqual(sum(p.numel() for p in self.models['NoGraph'].parameters()),48)
    def test_StandardGAT_50_parameters(self): self.assertEqual(sum(p.numel() for p in self.models['StandardGAT'].parameters()),50)
    def test_EpistemicGAT_50_parameters(self): self.assertEqual(sum(p.numel() for p in self.models['EpistemicGAT'].parameters()),50)

    def test_cloned_initialization_all_seeds(self):
        for seed in SEEDS:
            models=paired_models(seed,self.fixture)
            for a,b in zip(models['StandardGAT'].parameters(),models['EpistemicGAT'].parameters()):
                self.assertTrue(torch.equal(a,b));self.assertNotEqual(a.data_ptr(),b.data_ptr())

    def test_five_paired_seeds(self):
        self.assertEqual(len(seed_plan()),15);self.assertEqual(SEEDS,(101,202,303,404,505))

    def test_Adam_lr_weight_decay(self):
        optimizer=optimizer_for(self.models['NoGraph']);self.assertIsInstance(optimizer,torch.optim.Adam)
        self.assertEqual(optimizer.defaults['lr'],.001);self.assertEqual(optimizer.defaults['weight_decay'],.0001)

    def test_scenario_equal_BCE(self):
        ids=['fixture/val-%02d'%i for i in range(15)]
        rows=[ids[0]]*20+ids[1:];q=np.array([.9]*20+[.1]*14);y=np.ones(34)
        expected=(-np.log(.9)+14*(-np.log(.1)))/15
        self.assertAlmostEqual(scenario_macro_bce(q,y,rows,ids),expected,places=14)
        self.assertNotAlmostEqual(scenario_macro_bce(q,y,rows,ids),bce_samples(q,y).mean())

    def test_tracker_strict_min_delta(self):
        s=Selection();s.observe(0,1.);s.observe(1,1.-1e-5)
        self.assertEqual(s.bad,1);self.assertEqual(s.tracked_best,1.)
        s.observe(2,1.-1.1e-5);self.assertEqual(s.bad,0)

    def test_actual_lowest_independent_checkpoint(self):
        s=Selection();s.observe(0,1.);s.observe(1,.999999)
        self.assertEqual(s.tracked_best,1.);self.assertEqual(s.selected_epoch,1);self.assertEqual(s.bad,1)

    def test_earliest_checkpoint_tie(self):
        s=Selection();s.observe(0,.5);s.observe(1,.5);self.assertEqual(s.selected_epoch,0)

    def test_patience_ten(self):
        s=Selection();s.observe(0,.5)
        for e in range(1,11):
            _,stop=s.observe(e,.5);self.assertEqual(stop,e==10)

    def test_PCG64_paired_epoch_shuffle(self):
        for seed in SEEDS:
            for epoch in range(3):
                expected=np.random.Generator(np.random.PCG64(seed+epoch)).permutation(23)
                for method in METHODS: np.testing.assert_array_equal(epoch_permutation(23,seed,epoch),expected)

    def threshold_fixture(self,q,y):
        ids=['fixture/val-%02d'%i for i in range(15)]
        return select_threshold(np.tile(q,15),np.tile(y,15),np.repeat(ids,len(q)),ids)

    def test_threshold_candidate_set(self):
        result=self.threshold_fixture([.2,.8],[0,1]);self.assertEqual(result['candidates'],[0.,1-.8,1-.2,1.])

    def test_threshold_macro_F1_objective(self):
        ids=['fixture/val-%02d'%i for i in range(15)]
        q=[.3]*20+[.8]*14;y=[0]*20+[1]*14;scenario=[ids[0]]*20+ids[1:]
        result=select_threshold(q,y,scenario,ids)
        self.assertAlmostEqual(result['macro_corruption_F1'],1/15)

    def test_largest_threshold_exact_tie(self):
        result=self.threshold_fixture([.5,.5],[1,1]);self.assertEqual(result['tau'],1.)

    def test_zero_denominator_F1(self): self.assertEqual(corruption_f1([False],[False]),0.)

    def test_exact_existing_sender_prior(self):
        values=[0.,1e-12,.5]
        prior=compute_epistemic_weights(SimpleNamespace(use_epistemic_prior=True),dict(zip(NODE_ORDER,values)),list(NODE_ORDER))
        expected=np.asarray([[1/(1+e) for e in values]]*3,dtype=np.float32)
        np.testing.assert_array_equal(prior,expected);self.assertEqual(prior[0,0],1.)
        np.testing.assert_array_equal(self.models['EpistemicGAT'].prior(torch.tensor([values],dtype=torch.float64)).numpy()[0],prior)

    def test_exact_generic_layer_against_equations(self):
        model=self.models['EpistemicGAT'];model.eval()
        x=torch.tensor(self.fixture.nodes[:2],dtype=torch.float32)
        e=torch.tensor(self.fixture.nodes[:2,:,1],dtype=torch.float64);prior=model.prior(e)
        h=x
        for i,layer in enumerate(model.layers):
            batched,alpha=model.layer_forward(layer,h,prior,i==1)
            for j in range(2):
                generic,generic_alpha=layer(h[j],model.adjacency,prior[j],final_layer=i==1)
                torch.testing.assert_close(batched[j],generic,rtol=1e-6,atol=1e-7)
                torch.testing.assert_close(alpha[j],generic_alpha,rtol=1e-6,atol=1e-7)
                wh=layer.W(h[j]);pair=torch.cat((wh[:,None,:].expand(-1,3,-1),wh[None,:,:].expand(3,-1,-1)),dim=-1)
                logits=layer.leaky_relu(layer.a(pair).squeeze(-1))
                logits=logits+torch.log(prior[j]);masked=logits.masked_fill(model.adjacency==0,-1e9)
                expected=torch.softmax(masked,-1)*model.adjacency;expected=expected/expected.sum(-1,keepdim=True).clamp(min=1e-9)
                torch.testing.assert_close(generic_alpha,expected,rtol=0,atol=0)
            h=batched

    def test_zero_E_operators_identical(self):
        x=torch.tensor(self.fixture.nodes[:3],dtype=torch.float32);e=torch.zeros((3,3),dtype=torch.float64)
        a,b=self.models['StandardGAT'],self.models['EpistemicGAT'];a.eval();b.eval()
        torch.testing.assert_close(a(x,e),b(x,e),rtol=0,atol=0)

    def test_monotone_sender_no_receiver_factor(self):
        layer=GenericLayer(3,1,.1);layer.eval()
        with torch.no_grad(): layer.a.weight.zero_()
        h=torch.ones((3,3));adj=torch.tensor(ADJACENCY_LIST,dtype=torch.float32)
        zero=torch.ones((3,3));high=zero.clone();high[:,1]=.5
        _,base=layer(h,adj,zero,True);_,epi=layer(h,adj,high,True)
        self.assertLess(epi[0,1].item(),base[0,1].item());torch.testing.assert_close(epi[1],base[1],rtol=0,atol=0)

    def test_float32_and_no_message_passing_pool(self):
        m=self.models['NoGraph'];m.eval();x=torch.tensor(self.fixture.nodes[:3],dtype=torch.float32)
        expected=torch.sigmoid(m.last(F.elu(m.first(x))).squeeze(-1)).mean(-1)
        torch.testing.assert_close(m(x),expected,rtol=0,atol=0)
        self.assertEqual(m(x).dtype,torch.float32)

    def test_complete_tiny_synthetic_training(self):
        result=train_synthetic(self.fixture,101)
        orders=[result[m]['epoch_row_orders'] for m in METHODS]
        self.assertEqual(orders[0],orders[1]);self.assertEqual(orders[1],orders[2])
        for m in METHODS:
            losses=result[m]['validation_losses'];self.assertEqual(result[m]['selected_epoch'],losses.index(min(losses)))
            self.assertFalse(result[m]['real_model_trained'])


class ModelAdversarial(unittest.TestCase):
    def test_no_best_seed_or_replacement(self):
        for seeds in ((101,),SEEDS+(606,),(101,202,303,404,606)):
            with self.assertRaises(PreparationError): seed_plan(seeds)
        with self.assertRaises(PreparationError): paired_models(606,model_fixture())

    def test_real_scenario_cannot_enter_fixture(self):
        with self.assertRaises(PreparationError): SyntheticGraphs(np.ones((1,3,3),dtype=np.float64)*.3,[1],
            ['Town01/scenario-1'],['fixture:a'],['GRAPH_TRAIN'])

    def test_validation_15_required(self):
        with self.assertRaises(PreparationError): scenario_macro_bce([.5],[1],['fixture/v'],['fixture/v'])

    def test_negative_or_scaled_E_rejected(self):
        model=paired_models(101,model_fixture())['EpistemicGAT']
        for values in (torch.tensor([[-1.,0,0]],dtype=torch.float64),torch.tensor([[0.,0,0]],dtype=torch.float32)):
            with self.assertRaises(PreparationError): model.prior(values)

    def test_synthetic_epoch_cap(self):
        with self.assertRaises(PreparationError): train_synthetic(model_fixture(),101,fixture_epochs=100)
