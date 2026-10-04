"""Offline preparation only. Copies sealed inputs and mechanically reuses v4 code."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
PREPARATION=Path(__file__).resolve().parent
ROOT=PREPARATION.parents[1]
HERE=ROOT/'reports/carla_final_evaluation_bundle_v1'
V4=ROOT/'reports/carla_final_conformal_cal_bundle_v4'
SPEC=ROOT/'reports/carla_final_evaluation_preregistration_v1'
COMPLETE=ROOT/'reports/carla_final_conformal_cal_attempt003_complete_v1'
EVIDENCE=COMPLETE/'evidence/cognix_final_conformal_cal_attempt003_v2'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,obj): p.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def source(s,name):
    node=next(n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return ast.get_source_segment(s,node)
def copy(src,rel):
    target=HERE/rel; target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): assert sha(src)==sha(target),'Existing asset differs: '+rel
    else: shutil.copyfile(src,target)
    assert sha(target)==sha(src)

def main():
    if (HERE/'BUNDLE_SHA256SUMS').exists():raise RuntimeError('SEALED_BUNDLE_MUST_NOT_BE_REBUILT')
    spec=importlib.util.spec_from_file_location('ingestion',COMPLETE/'ingest_attempt003.py')
    ingest=importlib.util.module_from_spec(spec);spec.loader.exec_module(ingest)
    ingest.seal_check(COMPLETE,'ATTEMPT003_COMPLETE_SHA256SUMS',exact=True)
    ingest.validate(EVIDENCE)
    ingest.seal_check(V4,'BUNDLE_SHA256SUMS','7cfa0339d64e4d0350e8dd5b0940913e68d95e4d0aaffb2108593a11f87e9a39',True,
      {'cognix_final_conformal_cal_kaggle_v4.zip','cognix_final_conformal_cal_kaggle_v4.zip.sha256'})
    sealed=ingest.seal_check(SPEC,'SHA256SUMS','e0f20f0b012eff901acd0e46884635cc904a5be88c72bbfcffe5bb0d0a1c2593',True)
    needed=['final_metrics_spec.json','seed_aggregation_spec.json','decision_semantics.json','frozen_method_bindings.json','subgroup_reporting_protocol.json']
    bindings={}
    for n in needed:
        copy(SPEC/n,n);bindings[n]={'source':str((SPEC/n).relative_to(ROOT).as_posix()),'sha256':sealed[n]}
    copy(SPEC/'SHA256SUMS','provenance/preregistered_SHA256SUMS')
    copy(SPEC/'SHA256SUMS.sha256','provenance/preregistered_SHA256SUMS.sha256')
    copies=['final_evaluation.txt','final_conformal_cal.txt','prior_exposed_cal.txt','historical_exclusions.txt','partition_binding.json',
      'header_helpers.py','frozen_runtime.py','model_registry.json','runtime_lock.json','runtime_requirements.txt',
      'label_semantics.json','segmentation_oov_amendment.json','segmentation_oov_amendment.md',
      'upstream/manifest.json','upstream/fitted_oneclass_parameters.npz',
      'evidence/inventory/scenario_inventory_sorted.txt','evidence/inventory/scenario_inventory.json',
      'evidence/inventory/historical_exclusions.json']
    copies += [p.relative_to(V4).as_posix() for directory in ('frozen_sources','states') for p in (V4/directory).rglob('*') if p.is_file()]
    v4bind={}
    for rel in copies:copy(V4/rel,rel);v4bind[rel]=sha(V4/rel)
    copy(V4/'BUNDLE_SHA256SUMS','provenance/v4_BUNDLE_SHA256SUMS')
    copy(V4/'BUNDLE_SHA256SUMS.sha256','provenance/v4_BUNDLE_SHA256SUMS.sha256')
    copy(V4/'runner.py','provenance/v4_runner.py.txt')
    copy(V4/'streaming.py','provenance/v4_streaming.py.txt')
    for n in ['conformal_thresholds.json','threshold_calculation_receipts.json']:
        copy(EVIDENCE/n,n)
    for n in ['attempt003_completion_record.json','attempt003_archive_binding.json','attempt003_threshold_binding.json','attempt003_result_validation.json']:
        copy(COMPLETE/n,'provenance/'+n)
    copy(EVIDENCE/'RESULT_SHA256SUMS','provenance/Attempt003_RESULT_SHA256SUMS')
    copy(EVIDENCE/'RESULT_SHA256SUMS.sha256','provenance/Attempt003_RESULT_SHA256SUMS.sha256')
    # No calibration scores, calibration fitting function or calibration runner is active.
    adapter=(V4/'calibration_adapter.py').read_text()
    (HERE/'label_adapter.py').write_text('"""Exact sealed official-label alignment function; no calibrator."""\n'+source(adapter,'labels_from_table')+'\n',encoding='utf-8',newline='\n')
    runner=(V4/'runner.py').read_text()
    oov=source(runner,'map_segmentation_oov_to_frozen_vocabulary');runtime=source(runner,'runtime_check')
    (HERE/'scientific_runtime.py').write_text('"""Byte-identical v4 OOV and runtime functions."""\nimport json\nfrom pathlib import Path\nHERE=Path(__file__).resolve().parent\n\n'+oov+'\n\n'+runtime+'\n',encoding='utf-8',newline='\n')
    cls=source(runner,'CalProcessor').replace('CalProcessor','EvalProcessor').replace('self.cal','self.evaluation')
    cls=cls.replace('models, agents, cal, output, device','models, agents, evaluation, output, device').replace('frozenset(cal)','frozenset(evaluation)')
    cls=cls.replace('NON_FRESH_CAL','NON_FINAL_EVAL').replace('TOTAL_CAL_TICK_MEMORY_CAP_EXCEEDED','TOTAL_EVAL_TICK_MEMORY_CAP_EXCEEDED')
    cls=cls.replace('from calibration_adapter import labels_from_table, tick_scores','from label_adapter import labels_from_table')
    cls=cls.replace('all_scores[scorer] = tick_scores(np.concatenate(probs), labels[1:])','all_scores[scorer] = np.concatenate(probs)')
    cls=cls[:cls.index('        self.completed[sid] = all_scores')]+'''        from output_store import record_scenario
        record = record_scenario(self.output, sid, labels[1:], all_scores)
        self.completed[sid] = record
        print(json.dumps({"eval_completed": len(self.completed), "elapsed_ticks": self.total_ticks}), flush=True)
'''
    imports='"""Mechanically reused v4 scientific path; changed only role gate and output sink."""\nimport io\nimport json\nimport re\nfrom scientific_runtime import map_segmentation_oov_to_frozen_vocabulary\n\n'
    (HERE/'scientific_processor.py').write_text(imports+cls,encoding='utf-8',newline='\n')
    stream=(V4/'streaming.py').read_text()
    # Exact role exchange: the scientific branch now accepts only EVAL.
    import tokenize, io
    mapping={'cal':'evaluation','evaluation':'cal','on_cal':'on_eval'}
    tokens=[]
    for t in tokenize.generate_tokens(io.StringIO(stream).readline):
        if t.type==tokenize.NAME and t.string in mapping:t=t._replace(string=mapping[t.string])
        tokens.append(t)
    stream=tokenize.untokenize(tokens).replace('FINAL_EVAL_OPAQUE_DISCARD','FRESH_CAL_OPAQUE_DISCARD').replace('FRESH_CAL"','FINAL_EVAL"')
    stream=stream.replace('eval_scenarios_decoded','fresh_cal_scenarios_decoded').replace('CAL_MEMBER_CAP_EXCEEDED','EVAL_MEMBER_CAP_EXCEEDED')
    stream=stream.replace('CAL-only','EVAL-only').replace('on_cal','on_eval').replace('CAL membership','FINAL_EVAL membership')
    stream=stream.replace('Final EVAL/prior-exposed CAL/historical-excluded','Fresh CAL/prior-exposed CAL/historical-excluded').replace('No EVAL child','No protected-role child')
    (HERE/'streaming.py').write_text(stream,encoding='utf-8',newline='\n')
    save(HERE/'protocol_bindings.json',{'HEAD':ingest.HEAD,'readiness':'READY_FOR_SEPARATELY_AUTHORIZED_FINAL_EVALUATION_ATTEMPT_001',
      'FINAL_EVALUATION_ATTEMPT':'001','CALIBRATION_SOURCE':'successful Attempt003 v4','expected_archive_size':91538225599,'expected_archive_sha256':ingest.OFFICIAL_SHA,
      'v4_manifest_sha256':sha(V4/'BUNDLE_SHA256SUMS'),'v4_zip_sha256':sha(V4/'cognix_final_conformal_cal_kaggle_v4.zip'),
      'frozen_spec_manifest_sha256':sha(SPEC/'SHA256SUMS'),'frozen_spec_bindings':bindings,'identical_v4_files':v4bind,
      'threshold_sha256':sha(EVIDENCE/'conformal_thresholds.json'),'threshold_receipts_sha256':sha(EVIDENCE/'threshold_calculation_receipts.json'),
      'completion_seal_sha256':sha(COMPLETE/'ATTEMPT003_COMPLETE_SHA256SUMS'),'attempt003_evidence_archive_sha256':ingest.ARCHIVE_SHA,
      'role_counts':{'FINAL_EVAL':400,'FRESH_CAL_OPAQUE_DISCARD':100,'PRIOR_EXPOSED_CAL_OPAQUE_DISCARD':125,'HISTORICAL_EXCLUDED_OPAQUE_DISCARD':2},
      'source_function_sha256':{n:hashlib.sha256(source(runner,n).encode()).hexdigest() for n in ['runtime_check','map_segmentation_oov_to_frozen_vocabulary']},
      'label_function_sha256':hashlib.sha256(source(adapter,'labels_from_table').encode()).hexdigest(),
      'training':False,'threshold_refit':False,'calibration_scores_active':False,'prediction_ensemble':False,'best_seed_selection':False,
      'preparation_TEST_requests':0,'preparation_TEST_payload_bytes':0,'final_evaluation_executed':False})
    save(HERE/'v4_amendment_effect_audit.json',{'v4_changed':['formal CAL/EVAL membership','CAL 125 ->100','EVAL 500 ->400','old125 retired','rank120 ->96'],
      'v4_did_not_change':['metric formulas','method contrasts','seed aggregation','checkpoints','development hard thresholds','score semantics','class order','scientific pipeline'],
      'compatibility':'All frozen final formulas operate on assigned scenario membership and observed eligible-tick counts. No hard-coded 500 denominator is part of the metric formulas.',
      'scientific_choice_required':False,'mechanical_denominator':400,'source_specs':bindings})
    notebook={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Frozen COGNIX final evaluation — Attempt001\n','Preparation is not TEST-access authorization. Keep execution false until independently authorized. Configure the exact locked runtime externally; no automatic install/download.']},
      {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['from pathlib import Path\n','import subprocess, sys\n','BUNDLE = Path("/kaggle/working/cognix_final_evaluation_bundle_v1")\n','OUTPUT = Path("/kaggle/working/cognix_final_evaluation_attempt001_v1")\n','EXECUTE_SEPARATELY_AUTHORIZED_FINAL_EVALUATION = False\n']},
      {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['# Offline preflight; includes exact GPU/runtime lock check, no TEST request\n','subprocess.run([sys.executable, "-B", str(BUNDLE / "runner.py"), "--preflight"], check=True)\n']},
      {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['# One-shot execution only after separate explicit authorization\n','if EXECUTE_SEPARATELY_AUTHORIZED_FINAL_EVALUATION:\n','    assert not OUTPUT.exists(), "Existing attempt: no retry/resume"\n','    subprocess.run([sys.executable, "-B", str(BUNDLE / "runner.py"), "--execute-separately-authorized-final-evaluation"], check=True)\n','else:\n','    print("Execution disabled. Separate authorization required.")\n']}],
      'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.13'}},'nbformat':4,'nbformat_minor':5}
    save(HERE/'cognix_final_evaluation_v1.ipynb',notebook)
    print(json.dumps({'status':'SEALED_INPUTS_COPIED','specs_verified':True,'calibration_score_dependency':False}))

if __name__=='__main__':main()
