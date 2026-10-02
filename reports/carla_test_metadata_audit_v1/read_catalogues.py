"""Read publisher directory/catalogue metadata, never archives or dataset members."""
import hashlib
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent

class Catalogue(HTMLParser):
    def __init__(self, headings_only=False):
        super().__init__()
        self.headings_only = headings_only
        self.heading = False
        self.skip = 0
        self.text = []
        self.links = []
    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style', 'svg']:
            self.skip += 1
        if tag in ['h1', 'h2', 'h3', 'h4']:
            self.heading = True
            self.text.append('\nHEADING ' + str(dict(attrs).get('id', '')) + ': ')
        if tag == 'a' and not self.headings_only:
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)
        if tag in ['p', 'br', 'tr', 'li', 'pre'] and not self.headings_only:
            self.text.append('\n')
    def handle_endtag(self, tag):
        if tag in ['script', 'style', 'svg']:
            self.skip = max(0, self.skip - 1)
        if tag in ['h1', 'h2', 'h3', 'h4']:
            self.heading = False
    def handle_data(self, data):
        if not self.skip and (not self.headings_only or self.heading):
            self.text.append(data)

def main():
    ledger_path = HERE / 'metadata_access_ledger.json'
    ledger = json.loads(ledger_path.read_text())
    requests = [('https://data.carlanomaly.de/v1/', False, 'official_server_directory'),
                ('https://carlanomaly.de/resources/', False, 'official_resource_catalogue'),
                ('https://carlanomaly.de/about/', False, 'official_provenance_documentation'),
                ('https://carlanomaly.de/dataset/', True, 'official_dataset_section_headings')]
    for url, headings_only, name in requests:
        entry = {'order': len(ledger['official_metadata_accesses']) + 2, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                 'source': url, 'operation': 'GET bounded public HTML; extract section headings only' if headings_only else 'GET bounded public HTML catalogue/provenance prose',
                 'file_path_member': url, 'fields_inspected': ['section names/IDs only'] if headings_only else ['catalogue links', 'generation/provenance documentation', 'published composition'],
                 'why_metadata': 'Publisher documentation/index only; no archive, sensor member, event annotation or label table', 'state': 'planned_before_access'}
        ledger['official_metadata_accesses'].append(entry)
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
        try:
            with urllib.request.urlopen(url, timeout=25) as response:
                data = response.read(2_000_001)
                assert len(data) <= 2_000_000
                assert 'text/html' in response.headers.get('Content-Type', '')
                entry['HTTP_status'] = response.status
            parser = Catalogue(headings_only)
            parser.feed(data.decode('utf-8'))
            result = {'source': url, 'headings_only': headings_only, 'text': ''.join(parser.text), 'links': parser.links}
            (HERE / (name + '.json')).write_text(json.dumps(result, indent=2) + '\n')
            entry.update(state='completed', bytes_transferred=len(data), raw_document_sha256=hashlib.sha256(data).hexdigest(),
                         result_sha256=hashlib.sha256((HERE / (name + '.json')).read_bytes()).hexdigest(),
                         inspection_limit='No non-heading text retained/interpreted' if headings_only else 'Documentation text only')
            print(json.dumps(result), flush=True)
        except urllib.error.HTTPError as error:
            entry.update(state='failed', HTTP_status=error.code, response_body_read=False)
            print(json.dumps({'source': url, 'HTTP_status': error.code}), flush=True)
        ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')

if __name__ == '__main__':
    main()
