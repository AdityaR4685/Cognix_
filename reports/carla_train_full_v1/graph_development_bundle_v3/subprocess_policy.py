"""Exact sealed Gate-1 authenticity reads, not a general Git or shell capability."""
import functools
import shlex
import shutil
import subprocess
from pathlib import PurePosixPath
from graph_common import *

V6_SEAL='286b933c2d9b192d2a5a3a4797b51005bd42b16c65f4d8d2bacfcdaead086aa1'
READ_ONLY_ARGS=(('rev-parse','HEAD'),('status','--porcelain=v1','--untracked-files=normal'),
                ('diff','--name-only'),('diff','--cached','--name-only'),
                ('show','HEAD:reports/carla_gat_preregistration_v1/protocol.json'))


def valid_binding_path(value):
    return type(value) is str and bool(value) and value.isascii() and '\\' not in value and ':' not in value and \
        not value.startswith(('/', '-')) and not any(c.isspace() for c in value) and \
        all(p not in ('','.','..') and not p.startswith('-') for p in value.split('/')) and \
        PurePosixPath(value).as_posix()==value


@functools.lru_cache(maxsize=1)
def authenticated_policy():
    # No subprocess is used to bootstrap permission. External seals authenticate the
    # exact data and unedited caller before its paths can enlarge the read capability.
    seals={}
    for name,expected in (('gate1_execution_bundle_v8',SEALS['gate1_execution_bundle_v8']),
                         ('gate1_execution_bundle_v7',SEALS['gate1_execution_bundle_v7']),
                         ('gate1_execution_bundle_v6',V6_SEAL),
                         ('gate2_execution_bundle_v3',SEALS['gate2_execution_bundle_v3'])):
        seals[name]=verify_seal(REPORT/name,expected)
    path=REPORT/'gate1_execution_bundle_v8/recovery_bindings.json'
    binding=read_json(path)
    require(binding['schema_version']==3 and binding['preregistration_commit']==HEAD and
        binding['scientific_protocol_changed'] is False and re.fullmatch('[0-9a-f]{40}',HEAD),
        'Unrecognized frozen scientific commit/binding')
    records=binding['source_files'];seen=set();commands=[]
    require(type(records) is list and len(records)==22,'Authenticated verification binding cardinality changed')
    for record in records:
        name=record['path']
        require(valid_binding_path(name) and name not in seen and
            record['EOL_equivalence_verified'] is True and all(re.fullmatch('[0-9a-f]{64}',record[k]) for k in
            ('working_file_sha256','committed_blob_sha256')),'Unsafe/ambiguous scientific binding')
        seen.add(name)
        require(hash_file(REPO/name)==record['working_file_sha256'],'Bound scientific working bytes changed')
        commands.append((HEAD,name,record['committed_blob_sha256'],record['working_file_sha256']))
    executable=safe_path(shutil.which('git'))
    require(executable.name.lower()=='git.exe','Expected installed Git executable unavailable')
    probe=safe_path(os.environ['ComSpec'])
    require(probe.name.lower()=='cmd.exe','Expected reviewed Windows version probe unavailable')
    return dict(schema='Experiment-2B-v2-authenticated-subprocess-policy',expected_commit=HEAD,
        binding_path=str(path),binding_sha256=hash_file(path),upstream_seals=seals,
        recovery_common_sha256=hash_file(REPORT/'gate1_execution_bundle_v8/recovery_common.py'),
        gate2_resolution_sha256=hash_file(REPORT/'gate2_execution_bundle_v3/gate2_resolution.py'),
        scientific_bindings=tuple(commands),git_executable=str(executable),git_executable_sha256=hash_file(executable),
        reviewed_windows_probe=('cmd.exe','/c','ver'),reviewed_windows_probe_executable=str(probe),
        reviewed_windows_probe_executable_sha256=hash_file(probe),existing_read_only_arguments=READ_ONLY_ARGS)


def command_tokens(command):
    if isinstance(command,(tuple,list)):
        return tuple(command) if command and all(type(v) is str for v in command) else None
    if type(command) is not str:return None
    try:tokens=tuple(v[1:-1] if v.startswith('"') and v.endswith('"') else v for v in shlex.split(command,posix=False))
    except ValueError:return None
    # Windows supplies a native string audit event even for Popen(list, shell=False).
    # Accept only the canonical encoding of the exact argv, never shell syntax.
    return tokens if tokens and subprocess.list2cmdline(tokens)==command else None


def git_executable_admitted(value,policy):
    return type(value) is str and (value.lower() in ('git','git.exe') or
        Path(value).is_absolute() and norm(value)==norm(policy['git_executable']))


def subprocess_admission(audit_args,policy):
    if not isinstance(audit_args,(tuple,list)) or len(audit_args)!=4:return None
    executable,command,cwd,environment=audit_args
    # Preserve precisely Python's previously reviewed local version probe.
    # shell=True wraps this literal as /c "ver"; no other shell text is admitted.
    probe=policy['reviewed_windows_probe_executable']
    if type(command) is str and type(executable) is str and norm(executable)==norm(probe) and \
        command in (probe+' /c "ver"',probe+' /c ver'):
        return dict(kind='existing-reviewed-local-Windows-version-probe',argv=[probe,'/c','ver'])
    argv=command_tokens(command)
    if argv is None:return None
    # Native Windows audit events use executable=None for ordinary Popen(argv).
    # In that case CreateProcess selects argv[0]; an explicit foreign override fails.
    if git_executable_admitted(argv[0],policy) and (executable is None or git_executable_admitted(executable,policy)):
        if environment is not None and not isinstance(environment,dict):return None
        effective_environment=os.environ if environment is None else environment
        found=shutil.which(argv[0] if executable is None else executable,path=effective_environment.get('PATH'))
        if found is None or norm(found)!=norm(policy['git_executable']):return None
        if cwd is None or norm(cwd)!=norm(REPO):return None
        args=argv[1:]
        if args in READ_ONLY_ARGS:return dict(kind='existing-read-only-git',argv=list(argv))
        if len(args)!=2 or args[0]!='show':return None
        # This is strict membership in the sealed 22 (full commit,path) pairs.
        # HEAD, refs, alternate commits, extra options and arbitrary paths cannot match.
        for commit,path,blob_sha,working_sha in policy['scientific_bindings']:
            if args[1]==commit+':'+path:
                return dict(kind='authenticated-scientific-git-show',commit=commit,path=path,
                    expected_committed_blob_sha256=blob_sha,argv=list(argv))
        return None
    if len(argv)==3 and tuple(argv[1:])==('/c','ver') and \
        (argv[0].lower()=='cmd.exe' or Path(argv[0]).is_absolute() and norm(argv[0])==norm(probe)) and \
        (executable is None or executable.lower()=='cmd.exe' or Path(executable).is_absolute() and norm(executable)==norm(probe)):
        return dict(kind='existing-reviewed-local-Windows-version-probe',argv=list(argv))
    return None


def policy_report():
    policy=authenticated_policy()
    return dict(policy=policy,newly_admitted_commands=[dict(argv=['git','show',commit+':'+path],
        expected_committed_blob_sha256=blob,working_file_sha256=working,
        justification='Exact sealed recovery_bindings.source_files record required by unchanged recovery_common.verify_science loop')
        for commit,path,blob,working in policy['scientific_bindings']],
        blocked_in_v1_commands=[['git','show',commit+':'+path] for commit,path,_,_ in policy['scientific_bindings']],
        generic_git_show_allowed=False,other_git_commands_allowed=False,shell_commands_allowed=False,
        network_subprocesses_allowed=False,strict_cwd=str(REPO),
        native_Windows_command_encoding='Only subprocess.list2cmdline exact argv encoding; no shell invocation',
        output_admission='Unchanged Gate1 verifier compares every committed blob SHA256 with its sealed expected hash')
