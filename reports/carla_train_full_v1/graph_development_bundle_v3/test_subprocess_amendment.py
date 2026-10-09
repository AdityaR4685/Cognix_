import unittest
import subprocess
import graph_common
from graph_common import *
from subprocess_policy import *
from amendment_evidence import scientific_invariance,old_common
from current_upstream import authenticate_gate2


class SubprocessSynthetic(unittest.TestCase):
    def test_authenticated_Gate1_verify_science_path(self):
        modules=authenticate_gate2()['gate2_resolution'].authenticate_verifiers()
        result=modules[0].verify_science()
        self.assertEqual(result['preregistration_commit'],HEAD)
        self.assertEqual(len(result['source_files']),22)

    def test_all_22_exact_approved_git_show_reads(self):
        for commit,path,expected,working in authenticated_policy()['scientific_bindings']:
            raw=subprocess.check_output(['git','show',commit+':'+path],cwd=REPO)
            self.assertEqual(digest(raw),expected)

    def test_previously_failing_real_context_readonly(self):
        # Authenticate/validate existing evidence only: no raw replay, scoring or restore.
        reached={};previous=sys.getprofile()
        def observe(frame,event,arg):
            if event!='call':return
            path=Path(frame.f_code.co_filename)
            if not path.is_absolute():return
            if not any(path.is_relative_to(REPORT/name) for name in ('gate1_execution_bundle_v8','gate2_execution_bundle_v3')):return
            key=(path.relative_to(REPO).as_posix(),frame.f_code.co_qualname,frame.f_code.co_firstlineno)
            reached[key]=reached.get(key,0)+1
        sys.setprofile(observe)
        try:ctx=authenticate_gate2()['gate2_resolution'].real_context()
        finally:sys.setprofile(previous)
        self.assertEqual(len(ctx.scenarios),101);self.assertEqual(len(ctx.membership['FIT_NORMAL']),76)
        self.assertEqual(len(ctx.membership['CAL_NORMAL']),25)
        ctx.unchanged()
        attempt=Path(os.environ['TEMP']).name.rsplit('_',1)[1]
        write_json(BUNDLE/f'upstream_verification_call_trace_{attempt}.json',dict(
            scope='Exact unedited Gate2 real_context and reached Gate1/Gate2 functions; existing evidence validation only',
            reached_functions=[dict(path=p,qualified_function=fn,first_line=line,calls=count,
                source_file_sha256=hash_file(REPO/p)) for (p,fn,line),count in sorted(reached.items())],
            real_source_payload_replayed=False,features_recomputed=False,current_states_restored=False,
            current_models_initialized_or_trained=False,TEST_requests=0,network_requests=0))

    def test_v1_blocks_all_new_pairs(self):
        old=old_common();guard,counters=old.make_guard('preflight')
        for commit,path,_,_ in authenticated_policy()['scientific_bindings']:
            with self.assertRaises(old.PreparationError):guard('subprocess.Popen',('git',['git','show',commit+':'+path],str(REPO),None))
        self.assertEqual(counters['blocked_subprocess_attempts'],22)

    def test_only_v2_runtime_constants_and_token(self):
        self.assertEqual(DATA,REPORT/'graph_development_data_v3');self.assertEqual(RUNS,REPORT/'graph_development_runs_v3')
        self.assertEqual(PENDING,REPORT/'.graph_development_data_v3.pending')
        self.assertEqual(DATA_TOKEN,'HUMAN_REVIEWED_EXPERIMENT_2B_FULL_TRAIN_GRAPH_DATA_V3')
        future_namespaces_absent();self.assertEqual(verify_v1_failure()['units_count'],0)

    def test_scientific_whole_module_AST_invariance(self):
        report=scientific_invariance();self.assertEqual(report['status'],'PASS')
        self.assertTrue(report['graph_pair_identity_semantics_unchanged'])

    def test_kept_read_only_git_forms(self):
        policy=authenticated_policy()
        for args in READ_ONLY_ARGS:
            self.assertIsNotNone(subprocess_admission(('git',['git',*args],str(REPO),None),policy))

    def test_native_Windows_encoding_of_allowed_command(self):
        policy=authenticated_policy();commit,path,_,_=policy['scientific_bindings'][0]
        argv=['git','show',commit+':'+path]
        self.assertEqual(command_tokens(subprocess.list2cmdline(argv)),tuple(argv))
        self.assertIsNotNone(subprocess_admission(('git',subprocess.list2cmdline(argv),str(REPO),None),policy))
        self.assertIsNotNone(subprocess_admission((None,subprocess.list2cmdline(argv),str(REPO),None),policy))

    def test_real_activity_and_model_startup_remain_blocked(self):
        self.assertEqual(graph_common.ACTIVE_PHASE,'synthetic');self.assertEqual(ZERO['TEST_requests'],0)
        from current_restore import restore_current
        from graph_replay import FITSink
        with self.assertRaises(PreparationError):restore_current(None)
        with self.assertRaises(PreparationError):FITSink(None,None,None,None,None,None,None)

    def test_only_reviewed_Windows_version_probe_encodings(self):
        policy=authenticated_policy();probe=policy['reviewed_windows_probe_executable']
        for command in (probe+' /c "ver"',probe+' /c ver'):
            self.assertIsNotNone(subprocess_admission((probe,command,None,None),policy))
        for command in (probe+' /c "ver & calc"',probe+' /c "git show HEAD"',probe+' /c "ver" & calc'):
            self.assertIsNone(subprocess_admission((probe,command,None,None),policy))


class SubprocessAdversarial(unittest.TestCase):
    def denied(self,argv,cwd=None,executable=None):
        guard,counters=make_guard('synthetic')
        with self.assertRaises(PreparationError):
            guard('subprocess.Popen',(executable or argv[0],argv,str(REPO) if cwd is None else cwd,None))
        self.assertEqual(counters['blocked_subprocess_attempts'],1)

    def test_arbitrary_git_show_denied(self):
        for ref in ('HEAD','HEAD:pyproject.toml',HEAD+':pyproject.toml',HEAD+':cognix/adapters/carla/graph_fit_export.py'):
            self.denied(['git','show',ref])

    def test_wrong_commit_and_alternate_revision_denied(self):
        path=authenticated_policy()['scientific_bindings'][0][1]
        for ref in ('0'*40,'HEAD','main','HEAD~1',HEAD+'^','f3f876119b0d21655b3ba5c7db7bd996e628b2a2'):
            self.denied(['git','show',ref+':'+path])

    def test_wrong_or_user_supplied_scientific_path_denied(self):
        for path in ('cognix/core/types.py','cognix/adapters/carla/does_not_exist.py','reports/carla_gat_paired_execution_v1/results.json'):
            self.denied(['git','show',HEAD+':'+path])

    def test_path_traversal_and_alias_denied(self):
        for path in ('../cognix/adapters/carla/cache_builder.py','cognix/../cognix/adapters/carla/cache_builder.py',
                     'cognix\\adapters\\carla\\cache_builder.py','/cognix/adapters/carla/cache_builder.py'):
            self.denied(['git','show',HEAD+':'+path])
            self.assertFalse(valid_binding_path(path))

    def test_options_and_injections_denied(self):
        ref=HEAD+':'+authenticated_policy()['scientific_bindings'][0][1]
        for argv in (['git','--git-dir=elsewhere','show',ref],['git','show','--textconv',ref],
            ['git','show',ref,'--'],['git','show',ref+';git push'],['git','show',ref+'& calc'],
            ['git','show',ref+'\nfetch'],['git','-c','core.pager=cmd','show',ref]):self.denied(argv)

    def test_Git_write_and_network_commands_denied(self):
        for command in ('fetch','pull','push','add','commit','checkout','reset','merge','rebase','clean','tag'):
            self.denied(['git',command])

    def test_general_subprocess_and_shell_denied(self):
        for argv in (['python','-c','pass'],['powershell','-Command','git show HEAD'],
            ['cmd.exe','/c','git show HEAD'],['cmd.exe','/c','ver & calc'],['curl','https://example.invalid']):self.denied(argv)

    def test_spoofed_git_executable_denied(self):
        ref=HEAD+':'+authenticated_policy()['scientific_bindings'][0][1]
        self.denied([r'C:\untrusted\git.exe','show',ref])
        self.denied(['git','show',ref],executable='cmd.exe')

    def test_wrong_repository_cwd_denied(self):
        ref=HEAD+':'+authenticated_policy()['scientific_bindings'][0][1]
        self.denied(['git','show',ref],cwd=str(BUNDLE))

    def test_noncanonical_native_string_denied(self):
        policy=authenticated_policy();commit,path,_,_=policy['scientific_bindings'][0]
        for command in ('git  show '+commit+':'+path,'git show "'+commit+':'+path+'"',
                        'git show '+commit+':'+path+' | powershell','git show '+commit+':'+path+'\n'):
            self.assertIsNone(subprocess_admission(('git',command,str(REPO),None),policy))

    def test_v1_pending_write_delete_rename_blocked(self):
        before=verify_v1_failure();guard,c=make_guard('graph-data')
        for event,args in (('open',(str(V1_PENDING/'authorization.json'),'ab',os.O_WRONLY|os.O_APPEND)),
            ('open',(str(V1_PENDING/'units/new.json'),'xb',os.O_WRONLY|os.O_CREAT)),
            ('os.remove',(str(V1_PENDING/'authorization.json'),-1)),('os.rmdir',(str(V1_PENDING/'units'),-1)),
            ('os.rename',(str(V1_PENDING),str(PENDING),-1,-1)),
            ('os.rename',(str(PENDING),str(V1_PENDING),-1,-1)),('os.chmod',(str(V1_PENDING/'authorization.json'),0o600,-1))):
            with self.assertRaises(PreparationError):guard(event,args)
        self.assertEqual(before,verify_v1_failure())

    def test_v1_data_run_namespaces_never_writable(self):
        for phase in ('prepare','synthetic','preflight','graph-data'):
            guard,c=make_guard(phase)
            for name in ('graph_development_data_v1','graph_development_runs_v1','.graph_development_data_v1.pending'):
                with self.assertRaises(PreparationError):guard('open',(str(REPORT/name/'x'),'wb',os.O_WRONLY|os.O_CREAT))

    def test_v2_runtime_write_scope_exact(self):
        guard,c=make_guard('graph-data')
        for root in (DATA,PENDING):guard('open',(str(root/'x'),'wb',os.O_WRONLY|os.O_CREAT))
        for root in (RUNS,BUNDLE,REPORT/'graph_development_data_v4'):
            with self.assertRaises(PreparationError):guard('open',(str(root/'x'),'wb',os.O_WRONLY|os.O_CREAT))
        # Simulated audit events above perform no filesystem operations.
        self.assertFalse(DATA.exists());self.assertFalse(PENDING.exists());self.assertFalse(RUNS.exists())

    def test_network_TEST_and_preparation_payload_blocked(self):
        guard,c=make_guard('prepare')
        for event,args in (('socket.connect',(None,)),('socket.__new__',(None,)),
            ('open',(r'E:\carlanomaly-base-test.tar.gz','rb',os.O_RDONLY)),
            ('open',(SOURCE['path'],'rb',os.O_RDONLY))):
            with self.assertRaises(PreparationError):guard(event,args)
        self.assertEqual(c['TEST_requests'],0);self.assertEqual(c['network_requests'],0)
