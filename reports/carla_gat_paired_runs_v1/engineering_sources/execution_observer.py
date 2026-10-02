"""Read-only frame observer: pairing gates, progress, and retained final state.

The prepared independent launcher invokes the unchanged train.py. No scientific
function is replaced. Checks and extra evidence consume no random draws.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def install():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise RuntimeError('Observer refuses pre-import GPU/CUBLAS settings')
    if 'torch' in sys.modules:
        raise RuntimeError('Observer requires mask verification before PyTorch import')
    import torch
    import numpy as np
    from cognix.adapters.carla import graph_training as tr, graph_training_models as gm
    from cognix.adapters.carla import graph_training_data as gd
    method = sys.argv[sys.argv.index('--method') + 1]
    seed = int(sys.argv[sys.argv.index('--seed') + 1])
    base = Path(sys.argv[sys.argv.index('--output') + 1])
    evidence = Path(os.environ['COGNIX_OBSERVER_ROOT']) / (method + '_' + str(seed))
    evidence.mkdir(parents=True, exist_ok=False)
    started = time.time()
    tr.configure_determinism(seed)
    expected_rng = tr.recursive_content_hash(tr.rng_state())
    expected_pair = gm.paired_models(seed)
    expected_initial = {name: tr.recursive_content_hash({k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
                        for name, model in expected_pair.items()}
    del expected_pair
    state = {'actual_clone_checked': False, 'first_forward_checked': False, 'history_count': 0}
    lines, first = inspect.getsourcelines(tr.train_model)
    def line_number(text):
        return first + next(i for i, line in enumerate(lines) if line.strip() == text)
    forward_line = line_number('q = model(x, e)')
    history_line = line_number('if save:')
    final_line = line_number('changed = any(not torch.equal(initial_state[k], final_state[k]) for k in initial_state)')
    write(evidence/'launch_metadata.json', {'method':method, 'seed':seed, 'argv':sys.argv,
          'environment':tr.environment_record(), 'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES'],
          'verified_before_torch_import':True, 'started_unix':started, 'observer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def profile(frame, event, arg):
        if frame.f_code == gm.paired_models.__code__ and event == 'return' and frame.f_back.f_code == tr.train_model.__code__:
            standard, epi = arg['standard_gat'], arg['epistemic_gat']
            equal = all(torch.equal(v, epi.state_dict()[k]) for k,v in standard.state_dict().items())
            independent = all(v.data_ptr() != epi.state_dict()[k].data_ptr() for k,v in standard.state_dict().items())
            if not equal or not independent:
                raise RuntimeError('Actual trainer pair is not an independent byte-identical explicit clone')
            state['actual_clone_checked'] = True
            state['clone_parameters'] = {'explicit_clone':'frozen copy.deepcopy + load_state_dict',
                     'identical_tensors':equal, 'independent_storage':independent,
                     'standard_initial_hash':tr.recursive_content_hash(standard.state_dict()),
                     'epistemic_initial_hash':tr.recursive_content_hash(epi.state_dict())}

    def local_trace(frame, event, arg):
        if event == 'line':
            v = frame.f_locals
            if frame.f_lineno == forward_line and not state['first_forward_checked']:
                if not state['actual_clone_checked']:
                    raise RuntimeError('Actual clone verification missing before optimization')
                dataset = v['dataset']
                permutations = []
                for epoch in range(100):
                    a = list(gd.epoch_batches(dataset, seed, epoch))
                    b = list(gd.epoch_batches(dataset, seed, epoch))
                    if len(a) != len(b) or not all(np.array_equal(x,y) for x,y in zip(a,b)):
                        raise RuntimeError('Paired epoch permutations/boundaries differ')
                    order = np.concatenate(a)
                    permutations.append({'epoch':epoch+1,'row_order_sha256':hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
                                         'batch_sizes':[len(x) for x in a]})
                initial = tr.recursive_content_hash(v['initial_state'])
                rng = tr.recursive_content_hash(tr.rng_state())
                opt = v['optimizer'].state_dict()
                settings = [{k:value for k,value in group.items() if k!='params'} for group in opt['param_groups']]
                expected_opt = torch.optim.Adam(v['model'].parameters(), lr=.001, weight_decay=.0001).state_dict()
                if initial != expected_initial[method] or rng != expected_rng or opt != expected_opt or opt['state']:
                    raise RuntimeError('Actual first-forward initialization/dropout RNG/optimizer gate failed')
                if not np.array_equal(v['rows'], next(gd.epoch_batches(dataset, seed, 0))):
                    raise RuntimeError('Actual first batch differs from frozen plan')
                write(evidence/'paired_integrity.json', {'passed':True,'method':method,'seed':seed,
                       'verified_before_optimization':True,'clone':state['clone_parameters'],
                       'actual_initial_content_sha256':initial,'dropout_rng_sha256':rng,
                       'expected_seed_dropout_rng_sha256':expected_rng,'optimizer_settings':settings,
                       'optimizer_initial_state_empty':True,'all_100_epoch_permutations_and_boundaries_identical':True,
                       'epoch_plan':permutations,'frozen_training_specification':tr.TRAINING_SPEC})
                state['first_forward_checked'] = True
                sys.setprofile(None)
            elif frame.f_lineno == history_line:
                history = v['history']
                if len(history) != state['history_count']:
                    record = history[-1]
                    plan = json.loads((evidence/'paired_integrity.json').read_text())['epoch_plan'][record['epoch']-1]
                    if record['train_row_order_sha256'] != plan['row_order_sha256'] or record['batch_sizes'] != plan['batch_sizes']:
                        raise RuntimeError('Actual epoch permutation/boundaries differ from preoptimization gate')
                    write(evidence/'progress.json', {'method':method,'seed':seed,'epochs_completed':len(history),
                          'elapsed_seconds':time.time()-started,'last_epoch':record['epoch']}, exclusive=False)
                    state['history_count'] = len(history)
            elif frame.f_lineno == final_line:
                payload = {'model_state':v['final_state'],'optimizer_state':v['optimizer'].state_dict(),
                           'rng_state':tr.rng_state(),'epoch':v['epoch'],'method':method,'seed':seed}
                with (evidence/'observer_final_state.pt').open('xb') as stream:
                    torch.save(payload, stream)
                write(evidence/'final_state.json', {'content_sha256':tr.recursive_content_hash(payload),
                      'role':'last executed epoch state; does not change selected checkpoint',
                      'epoch':v['epoch']})
        elif event == 'return':
            write(evidence/'observer_completion.json', {'passed':isinstance(arg,dict) and arg.get('status')=='complete',
                  'method':method,'seed':seed,'elapsed_seconds':time.time()-started,
                  'first_forward_checked':state['first_forward_checked'],
                  'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
                  'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(0),
                  'environment':tr.environment_record()})
            sys.settrace(None)
        return local_trace

    def global_trace(frame, event, arg):
        return local_trace if frame.f_code == tr.train_model.__code__ else None
    sys.setprofile(profile)
    sys.settrace(global_trace)
