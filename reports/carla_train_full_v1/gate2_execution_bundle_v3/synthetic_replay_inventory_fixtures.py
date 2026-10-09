"""Actual sealed builder generates tiny, valid production-schema inventory."""
import gzip
from gate2_common import write_new, hash_file
from gate2_replay_inventory import ReplayInventory


class CanonicalFixture:
    def __init__(self, root):
        import local_inventory
        from local_source import verify_source
        from test_clean_train import train_payloads, scenario_tar
        self.root, self.legacy = root, local_inventory
        self.source = root/'artificial-train.tar.gz'
        payloads = train_payloads(n=15)
        raw = scenario_tar(payloads,'Town01/scenario-1') + scenario_tar(payloads,'Town02/scenario-2') + bytes(1024)
        write_new(self.source,gzip.compress(raw,mtime=0))
        identity = {'path':str(self.source),'bytes':self.source.stat().st_size,'sha256':hash_file(self.source)}
        self.binding = verify_source(identity)
        self.directory = root/'inventory'
        self.index = local_inventory.build_inventory(self.source,self.binding,self.directory)
        self.records = list(local_inventory.inventory_records(self.directory,self.index))
        assert all('record_sha256' not in r for r in self.records)
        assert all(e == dict(local_inventory.summary(r),record_sha256=local_inventory.digest(r))
                   for e,r in zip(self.index['scenarios'],self.records))

    def admitted(self, files=()):
        return ReplayInventory(self.index,self.binding,self.legacy,files)
