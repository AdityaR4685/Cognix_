"""NOT RUN in preparation. Requires separately reviewed external seal and explicit data authorization."""
import argparse
from graph_common import *


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-seal',required=True)
    parser.add_argument('--authorize-graph-data-generation',required=True,choices=[DATA_TOKEN])
    args=parser.parse_args()
    counters=install_guard('graph-data',source_integrity=True)
    verify_seal(BUNDLE,args.bundle_seal)
    from execution_closure import pre_source_closure,complete_source_integrity,admission_then_publication
    from graph_replay import replay_current
    admission_then_publication(
        lambda:pre_source_closure(args.bundle_seal,progress=lambda message:print(message,flush=True)),
        lambda closure:complete_source_integrity(closure,lambda message:print(message,flush=True)),
        lambda closure,source:publish_authorized_data(args.bundle_seal,args.authorize_graph_data_generation,closure,source))


def publish_authorized_data(bundle_seal,authorization,closure,source):
    require(__import__('graph_common').ACTIVE_PHASE=='graph-data','Data publication guard missing')
    require(authorization==DATA_TOKEN and closure['status']==source['status']=='PASS' and
        source_fingerprint()==source['fingerprint_after'],'Publication admission changed')
    future_namespaces_absent()
    PENDING.mkdir();(PENDING/'units').mkdir()
    write_json(PENDING/'authorization.json',dict(bundle_seal=bundle_seal,authorization=authorization,
        scope='Graph data only; model execution is forbidden',HEAD=HEAD))
    write_json(PENDING/'pre_source_admission.json',dict(closure=closure,source_integrity=source))
    from graph_replay import replay_current
    manifest=replay_current()
    manifest['preparation_bundle_seal']=bundle_seal
    write_json(PENDING/'manifest.json',manifest)
    require(not RUNS.exists() and not DATA.exists(),'Publish destination or model runtime already exists')
    data_seal=seal_tree(PENDING,'artifact_inventory.json')
    # os.rename on Windows refuses an existing destination; never os.replace.
    PENDING.rename(DATA)
    for path in DATA.rglob('*'):
        if path.is_file(): path.chmod(stat.S_IREAD)
    print(json.dumps(dict(status='GRAPH_DATA_COMPLETE',data_path=str(DATA),data_seal=data_seal,
        graph_training_authorized=False,TEST_requests=0,network_requests=0)))


if __name__=='__main__': main()
