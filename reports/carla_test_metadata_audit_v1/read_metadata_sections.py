"""Select composition/directory documentation; discard annotation/sensor sections."""
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from read_catalogues import Catalogue

HERE = Path(__file__).resolve().parent
lp = HERE / 'metadata_access_ledger.json'
ledger = json.loads(lp.read_text())
sources = [('https://carlanomaly.de/dataset/', 'composition_and_directory_documentation'),
           ('https://data.carlanomaly.de/v1/SHA256SUMS.txt', 'official_archive_checksums')]
for url, name in sources:
    entry = {'order': len(ledger['official_metadata_accesses']) + 2,
             'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'source': url,
             'operation': 'Read only introductory composition and directory-structure documentation sections' if name.startswith('composition') else 'Read archive checksum catalogue',
             'file_path_member': url, 'fields_inspected': ['published scenario counts', 'split/town/condition/type path grammar', 'provenance statements'] if name.startswith('composition') else ['archive filename', 'sha256'],
             'why_metadata': 'Publisher catalogue/documentation; no actual scenario sensor/label/event payload', 'state': 'planned_before_access'}
    ledger['official_metadata_accesses'].append(entry)
    lp.write_text(json.dumps(ledger, indent=2) + '\n')
    with urllib.request.urlopen(url, timeout=25) as response:
        data = response.read(2_000_001)
        assert len(data) <= 2_000_000
        entry['HTTP_status'] = response.status
    decoded = data.decode()
    if name.startswith('composition'):
        heading = re.search(r'<h2\b[^>]*\bid=[\"\x27]?sensor-setup[\"\x27\s>]', decoded)
        assert heading
        intro = decoded[:heading.start()]
        start = re.search(r'<h2\b[^>]*\bid=[\"\x27]?directory-structure[\"\x27\s>]', decoded)
        assert start
        end = re.search(r'<h2\b', decoded[start.end():])
        assert end
        directory = decoded[start.start():start.end() + end.start()]
        result = {'source': url, 'sections': {}}
        for section, html in [('introduction_composition', intro), ('directory_structure', directory)]:
            parser = Catalogue()
            parser.feed(html)
            result['sections'][section] = ''.join(parser.text).strip()
        result['excluded_sections'] = ['sensor setup', 'anomaly descriptions', 'environmental conditions', 'all sensor/annotation/sample/timestep/scenario label examples', 'additional data values']
    else:
        result = {'source': url, 'text': decoded}
    destination = HERE / (name + '.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    entry.update(state='completed', bytes_transferred=len(data), raw_document_sha256=hashlib.sha256(data).hexdigest(),
                 result_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                 inspection_limit='Only selected metadata sections retained/interpreted; raw HTML is documentation, never a dataset payload')
    lp.write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(result), flush=True)
