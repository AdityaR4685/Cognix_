"""Workspace-only synthetic scratch directories with inherited Windows permissions."""
import shutil
import uuid
from pathlib import Path
from partition_common import REPO, require, safe_path


class SyntheticDirectory:
    def __init__(self, parent):
        parent = safe_path(parent)
        require(parent == REPO or parent.name.startswith('synthetic-partition-tests-'),
                'Invalid synthetic fixture parent')
        self.path = parent / ('synthetic-partition-tests-' + uuid.uuid4().hex)
        # Python 3.14 tempfile mode 0700 excludes the sandbox AppContainer on Windows.
        # Default mkdir permissions inherit the authorized workspace directory ACL.
        self.path.mkdir()
        self.name = str(self.path)

    def cleanup(self):
        resolved = self.path.resolve()
        require(resolved == self.path and resolved.is_relative_to(REPO) and
                resolved.name.startswith('synthetic-partition-tests-'), 'Unsafe synthetic cleanup target')
        if resolved.exists():
            shutil.rmtree(resolved)

    def __enter__(self):
        return self.name

    def __exit__(self, *args):
        self.cleanup()
