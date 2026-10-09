"""Exact v2 preservation, scientific byte/AST equality and explicit engineering normalization."""
import ast
import copy
import difflib
import importlib.util
from graph_common import *

ENGINEERING_MODULES={'preflight_graph.py','prepare_amendment.py','amendment_evidence.py'}
EXACT_DOCUMENTS=('current_experiment_bindings.json','current_partition_audit.json','current_gate2_restore_audit.json',
    'graph_science_spec.json','training_spec.json','threshold_spec.json','historical_protocol_adaptation.json',
    'historical_evidence_exclusion.json')
NAMESPACE_DOCUMENTS=('graph_export_spec.json','graph_training_authorization_design.json')


def old_common():
    spec=importlib.util.spec_from_file_location('_sealed_graph_common_v1',V1_BUNDLE/'graph_common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def ast_dump(node):return ast.dump(node,annotate_fields=True,include_attributes=False)


def namespace_text(raw):
    return raw.replace('graph_development_data_v2','graph_development_data_v3').replace(
        'graph_development_runs_v2','graph_development_runs_v3').replace('FULL_TRAIN_GRAPH_DATA_V2','FULL_TRAIN_GRAPH_DATA_V3')


def expected_common(old):
    tree=ast.parse(namespace_text(old.decode()))
    # Historic V1 evidence literals are untouched. Only the V3 future glob changes.
    extra=ast.parse("V2_BUNDLE = REPORT / 'graph_development_bundle_v2'\nV2_BUNDLE_SEAL = 'b0bb3ff9bdb33732442de9b8ea47b415039307612e13924f39c520bc02406c6a'").body
    index=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='V1_PENDING' for t in n.targets))+1
    tree.body[index:index]=extra
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='future_namespaces_absent')
    # Historic V2 paths are protected by absence assertions, never writable.
    fn.body.extend(ast.parse("for name in ('graph_development_data_v2','graph_development_runs_v2','.graph_development_data_v2.pending'):\n    require(not safe_path(REPORT/name).exists(),'Historical v2 runtime cannot be reused')").body)
    return tree


def verify_publication_science(old,new):
    a=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    b=next(n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='publish_authorized_data')
    start=next(i for i,n in enumerate(a.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
        isinstance(n.value.func,ast.Name) and n.value.func.id=='future_namespaces_absent')
    expected=copy.deepcopy(a.body[start:])
    class Parameters(ast.NodeTransformer):
        def visit_Attribute(self,n):
            if isinstance(n.value,ast.Name) and n.value.id=='args' and n.attr in ('bundle_seal','authorize_graph_data_generation'):
                return ast.Name(id='bundle_seal' if n.attr=='bundle_seal' else 'authorization',ctx=ast.Load())
            return self.generic_visit(n)
    expected=[Parameters().visit(n) for n in expected]
    auth_index=next(i for i,n in enumerate(expected) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
        isinstance(n.value.func,ast.Name) and n.value.func.id=='write_json')
    expected[auth_index+1:auth_index+1]=ast.parse("write_json(PENDING/'pre_source_admission.json',dict(closure=closure,source_integrity=source))").body
    require(ast_dump(ast.Module(body=expected,type_ignores=[]))==ast_dump(ast.Module(body=b.body[2:],type_ignores=[])),
        'Unexplained future replay/publication tail change')


def expected_test_runner(old):
    tree=ast.parse(old)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.ImportFrom) and n.module=='test_subprocess_amendment')+1
    main.body[index:index]=ast.parse('from test_production_closure import ProductionIntegration,OrderingAdversarial,write_test_reports').body
    loop=next(n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='cls')
    loop.iter.elts.extend([ast.Name(id='ProductionIntegration',ctx=ast.Load()),ast.Name(id='OrderingAdversarial',ctx=ast.Load())])
    index=next(i for i,n in enumerate(main.body) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='item')
    main.body[index:index]=ast.parse('write_test_reports(record)').body
    return tree


def scientific_invariance():
    verify_seal(V1_BUNDLE,V1_BUNDLE_SEAL);verify_seal(V2_BUNDLE,V2_BUNDLE_SEAL)
    records=[];exact=[]
    for path in sorted(V2_BUNDLE.glob('*.py')):
        name=path.name;old=path.read_bytes();new=(BUNDLE/name).read_bytes();a=ast.parse(old);b=ast.parse(new)
        if name in ENGINEERING_MODULES:
            equality='Engineering-only evidence/lifecycle/pre-source entrypoint; no scientific definitions'
        elif name=='graph_common.py':
            require(ast_dump(expected_common(old))==ast_dump(b),'Unexplained shared scientific constants/guard AST change')
            equality='Only V3 runtime/token and historical V2 absence bindings'
        elif name=='test_subprocess_amendment.py':
            expected=old.decode().replace("REPORT/'graph_development_data_v3'","REPORT/'graph_development_data_v4'")
            require(ast_dump(ast.parse(namespace_text(expected)))==ast_dump(b),'Unexplained retained subprocess test change')
            equality='Original 24 tests retained; V3 namespace assertions only'
        elif name=='generate_graph_data.py':
            verify_publication_science(old,new)
            equality='Shared pre-source closure before pending; unchanged source replay/publication tail'
        elif name=='run_synthetic_tests.py':
            require(ast_dump(expected_test_runner(old))==ast_dump(b),'Unexplained original test harness change')
            equality='Original 80 tests retained; additive classes and named report emission'
        else:
            require(old==new,'Scientific or subprocess-policy Python changed: '+name);exact.append(name)
            equality='Byte-identical whole module and every function/class AST'
        functions=[]
        for fn in a.body:
            if isinstance(fn,(ast.FunctionDef,ast.ClassDef)):
                matches=[n for n in b.body if isinstance(n,type(fn)) and n.name==fn.name]
                functions.append(dict(name=fn.name,v2_AST_sha256=digest(ast_dump(fn).encode()),
                    v3_AST_sha256=digest(ast_dump(matches[0]).encode()) if matches else None,
                    exact_AST_equal=bool(matches and ast_dump(fn)==ast_dump(matches[0]))))
        records.append(dict(file=name,v2_sha256=digest(old),v3_sha256=digest(new),byte_identical=old==new,
            permitted_engineering_normalization=equality,functions=functions))
    documents=[]
    for name in EXACT_DOCUMENTS+NAMESPACE_DOCUMENTS:
        old=(V2_BUNDLE/name).read_bytes();new=(BUNDLE/name).read_bytes()
        require(new==(namespace_text(old.decode()).encode() if name in NAMESPACE_DOCUMENTS else old),
            'Scientific document bytes changed: '+name)
        documents.append(dict(file=name,v2_sha256=digest(old),v3_sha256=digest(new),scientifically_equal=True,
            normalization='V3 runtime paths only' if name in NAMESPACE_DOCUMENTS else 'none'))
    require(hash_file(BUNDLE/'subprocess_policy.py')==hash_file(V2_BUNDLE/'subprocess_policy.py'),'Reviewed V2 subprocess policy changed')
    for name in ('v1_failure_evidence.json','v1_failure_readonly_reproduction.txt'):
        require((BUNDLE/name).read_bytes()==(V2_BUNDLE/name).read_bytes(),'V1 forensic evidence changed')
    return dict(status='PASS',v1_seal=V1_BUNDLE_SEAL,v2_seal=V2_BUNDLE_SEAL,python=records,documents=documents,
        scientific_Python_byte_identical=exact,scientific_files_changed=[],
        graph_pair_identity_semantics_unchanged=True,all_primary_graph_functions_AST_equal=True,
        current_restoration_and_upstream_science_byte_identical=True,subprocess_policy_byte_identical=True,
        allowed_normalizations=['V3 runtime paths/token and historical V2 absence bindings',
            'Shared engineering pre-source execution order and publication admission evidence',
            'Engineering evidence/lifecycle entrypoints','Additive tests; retained namespace assertions ported to V3'],
        membership_recomputed=False,no_scientific_parameter_or_behavior_changes=True)


def verify_amendment():
    failure=read_json(BUNDLE/'v1_failure_evidence.json')
    require(verify_v1_failure()==failure['preserved_pending'],'V1 pending evidence changed')
    from subprocess_policy import policy_report
    require(canonical(policy_report())==canonical(read_json(BUNDLE/'subprocess_allowlist_audit.json')),
        'Reviewed V2 subprocess admission policy changed')
    require(scientific_invariance()==read_json(BUNDLE/'scientific_invariance_audit.json'),'V3 scientific invariance changed')
    require(all(p.stat().st_file_attributes&stat.FILE_ATTRIBUTE_READONLY for p in V2_BUNDLE.rglob('*') if p.is_file()),
        'V2 immutable attributes changed')
    return dict(status='PASS',v1_pending_preserved=True,v1_bundle_preserved=True,v2_bundle_preserved=True)


def write_diff():
    lines=[];modified=[];new=[];unchanged=[]
    for p in sorted(BUNDLE.glob('*.py')):
        old=V2_BUNDLE/p.name
        if old.exists() and old.read_bytes()==p.read_bytes():unchanged.append(p.name);continue
        if old.exists():modified.append(p.name);before=old.read_text().splitlines(True)
        else:new.append(p.name);before=[]
        lines.extend(difflib.unified_diff(before,p.read_text().splitlines(True),fromfile=str(old) if old.exists() else '/dev/null',tofile=str(p)))
    write_new(BUNDLE/'engineering.diff',''.join(lines).encode())
    return dict(modified_python_relative_to_v2=modified,new_python_relative_to_v2=new,byte_identical_python=unchanged)
