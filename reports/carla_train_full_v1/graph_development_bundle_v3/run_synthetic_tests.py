"""Guarded CPU-only fixture evidence; writes unique attempt records, never replaces failed evidence."""
import argparse
import io
import unittest
import shutil
from graph_common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,required=True);args=parser.parse_args()
    require(args.attempt>=1,'Invalid attempt')
    scratch=BUNDLE/f'synthetic_scratch_{args.attempt}'
    require(not scratch.exists(),'Scratch attempt already exists');scratch.mkdir()
    os.environ.update(TEMP=str(scratch),TMP=str(scratch),TORCHINDUCTOR_CACHE_DIR=str(scratch/'torch_cache'),
                      OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    counters=install_guard('synthetic')
    from test_graph_export import ExportSynthetic,ExportAdversarial
    from test_graph_training import ModelSynthetic,ModelAdversarial
    from test_subprocess_amendment import SubprocessSynthetic,SubprocessAdversarial
    from test_production_closure import ProductionIntegration,OrderingAdversarial,write_test_reports
    suite=unittest.TestSuite()
    for cls in (ExportSynthetic,ExportAdversarial,ModelSynthetic,ModelAdversarial,SubprocessSynthetic,SubprocessAdversarial,ProductionIntegration,OrderingAdversarial):
        for name in sorted(cls.__dict__):
            if name.startswith('test_'):suite.addTest(cls(name))
    stream=io.StringIO()
    class Result(unittest.TextTestResult):
        records=[]
        def addSuccess(self,test): super().addSuccess(test);self.records.append(dict(test=test.id(),status='PASS'))
        def addFailure(self,test,err): super().addFailure(test,err);self.records.append(dict(test=test.id(),status='FAIL',detail=self._exc_info_to_string(err,test)))
        def addError(self,test,err): super().addError(test,err);self.records.append(dict(test=test.id(),status='ERROR',detail=self._exc_info_to_string(err,test)))
        def addSkip(self,test,reason): super().addSkip(test,reason);self.records.append(dict(test=test.id(),status='SKIP',detail=reason))
    result=unittest.TextTestRunner(stream=stream,verbosity=2,resultclass=Result).run(suite)
    import torch
    record=dict(attempt=args.attempt,evidence_kind='synthetic_fixtures_only',real_activity=dict(ZERO),
        passed=sum(r['status']=='PASS' for r in result.records),failed=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),tests_run=result.testsRun,records=result.records,guard_counters=counters,
        environment=dict(python=sys.version,numpy=__import__('numpy').__version__,torch=torch.__version__,
            device='CPU',cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),TF32=False,AMP=False),
        bundle_python_sha256={p.name:hash_file(p) for p in sorted(BUNDLE.glob('*.py'))})
    write_json(BUNDLE/f'synthetic_test_attempt_{args.attempt}.json',record)
    write_new(BUNDLE/f'synthetic_test_attempt_{args.attempt}.txt',stream.getvalue().encode())
    write_json(BUNDLE/f'adversarial_test_attempt_{args.attempt}.json',dict(attempt=args.attempt,
        evidence_kind='blocked synthetic adversarial attempts; zero actual TEST/network requests',
        records=[r for r in result.records if 'Adversarial.' in r['test']]))
    write_test_reports(record)
    for item in scratch.rglob('*'):safe_path(item)
    shutil.rmtree(scratch)
    print(json.dumps({k:record[k] for k in ('attempt','passed','failed','errors','skipped','tests_run')}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())
