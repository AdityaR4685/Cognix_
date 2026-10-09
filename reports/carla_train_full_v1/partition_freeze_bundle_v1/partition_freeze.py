"""Future explicitly authorized freeze; ends at DATA / INTEGRITY, before fitting."""
import argparse
import json
import sys
sys.dont_write_bytecode = True
from partition_common import (BUNDLE, TARGET, TOKEN, require, verify_manifest,
    make_guard, ReviewRequired)
from partition_evidence import verify_inputs
from partition_artifact import freeze
from partition_preflight import progress_printer


def make_freeze_guard():
    # Accept only the new runtime and its pending sibling, never historical writes.
    base_guard, counters = make_guard(allow_source=True)
    def guard(event, args):
        if event in ('open', 'os.mkdir', 'os.rename'):
            from pathlib import Path
            import os
            writes = event != 'open' or (isinstance(args[1], str) and any(c in args[1] for c in 'wax+')) or \
                bool(args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writes:
                values = args[:2] if event == 'os.rename' else args[:1]
                paths = [Path(os.fsdecode(v)).absolute() for v in values if isinstance(v, (str, bytes, os.PathLike))]
                def allowed(p):
                    return p == TARGET or TARGET in p.parents or any(
                        a.parent == TARGET.parent and a.name.startswith('.' + TARGET.name + '.pending-')
                        for a in (p, *p.parents))
                require(paths and all(allowed(p) for p in paths), 'Partition write escaped new runtime')
                return
        base_guard(event, args)
    return guard, counters


def run(authorization, bundle_seal):
    require(authorization == TOKEN, 'Explicit partition-freeze authorization token required')
    guard, counters = make_freeze_guard()
    sys.addaudithook(guard)
    verify_manifest(BUNDLE, expected_seal=bundle_seal)
    evidence = verify_inputs(progress_printer())
    result = freeze(evidence, TARGET, bundle_seal)
    result['guard_counters'] = counters
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorize-partition-freeze', required=True)
    parser.add_argument('--bundle-seal', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.authorize_partition_freeze, args.bundle_seal), indent=2, sort_keys=True))
    except (ReviewRequired, OSError, ValueError, KeyError, RuntimeError) as exc:
        print('STOP FOR HUMAN REVIEW: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
