"""Read-only real input verification. An absent partition is never computed here."""
import argparse
import json
import sys
import time
sys.dont_write_bytecode = True
from partition_common import BUNDLE, TARGET, verify_manifest, install_guard, ReviewRequired
from partition_evidence import verify_inputs, preflight_summary
from partition_artifact import check_target_namespace, verify_existing


def run(bundle_seal, progress=None):
    counters = install_guard(allow_source=True)
    verify_manifest(BUNDLE, expected_seal=bundle_seal)
    evidence = verify_inputs(progress)
    check_target_namespace(TARGET)
    target_state = 'absent'
    if TARGET.exists():
        verify_existing(TARGET, evidence, bundle_seal)
        target_state = 'present_verified_immutable'
    return preflight_summary(evidence, target_state, counters)


def progress_printer():
    last = [0.0]
    def progress(count):
        if time.monotonic() - last[0] >= 30:
            print('Read-only source SHA256 verification: %s bytes' % count, file=sys.stderr, flush=True)
            last[0] = time.monotonic()
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-seal', required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.bundle_seal, progress_printer()), indent=2, sort_keys=True))
    except (ReviewRequired, OSError, ValueError, KeyError, RuntimeError) as exc:
        print('STOP FOR HUMAN REVIEW: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
