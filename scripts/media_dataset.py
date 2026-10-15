"""Discover media tweets using Parquet range reads, then fetch selected full rows.

Run from repository root with Python 3.10+ and pyarrow installed.
Intermediate URL scans stay in --cache, outside the versioned dataset folder.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

DATASET = 'deadbirds/usc-x-24-us-election-parquet'
REVISION = 'd0d93f9a457e54d21cb6c51ec9340548fa77c9cc'
SOURCE_REVISION = 'd8fc6f7d2510d89c78f6175601055288f643a898'
CANDIDATES = ['nytimes', 'foxnews', 'reuters', 'ap', 'nbcnews', 'abc',
              'cbsnews', 'bbcworld', 'cnn', 'washingtonpost', 'usatoday', 'wsj']
URL_PATTERN = re.compile(r'https?://(?:www\.)?(?:twitter\.com|x\.com)/([^/]+)/status/(\d+)', re.I)


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


class RemoteParquet(io.RawIOBase):
    """Seekable HTTP range reader that refuses full-file responses."""
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.bytes_read = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = min(n if n >= 0 else self.size - self.pos, self.size - self.pos)
        if n <= 0: return b''
        start, end = self.pos, self.pos + n - 1
        request = urllib.request.Request(
            self.url + f'?start={start}&end={end}', headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError('Range request ignored; refusing full dataset-file download')
            data = response.read(n + 1)
        if len(data) != n:
            raise RuntimeError('Incomplete range response')
        self.pos += n
        self.bytes_read += n
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def retry(function, value):
    for attempt in range(4):
        try:
            return function(value)
        except Exception:
            if attempt == 3: raise
            time.sleep(2 ** attempt)


def scan_file(entry):
    url = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/default/train/{entry["filename"]}'
    remote = RemoteParquet(url, entry['size'])
    parquet = pq.ParquetFile(remote, pre_buffer=False, buffer_size=0)
    found = []
    for index, url in enumerate(parquet.read(columns=['url']).column('url').to_pylist()):
        match = URL_PATTERN.match(url or '')
        if match and match[1].lower() in CANDIDATES:
            found.append({'account': match[1].lower(), 'tweet_id': match[2], 'row': index})
    return {'rows': parquet.metadata.num_rows, 'bytes': remote.bytes_read, 'matches': found}


def discover(cache):
    cache.mkdir(parents=True, exist_ok=True)
    manifest = get_json('https://datasets-server.huggingface.co/parquet?' + urllib.parse.urlencode({'dataset': DATASET}))
    files = sorted(manifest['parquet_files'], key=lambda x: x['filename'])
    if manifest.get('partial') or manifest.get('failed') or manifest.get('pending'):
        raise RuntimeError('Dataset manifest is incomplete')
    def cached_scan(entry):
        path = cache / (entry['filename'] + '.json')
        if path.exists(): return json.loads(path.read_text())
        result = retry(scan_file, entry)
        path.write_text(json.dumps(result))
        return result
    found, offset = [], 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for number, (entry, result) in enumerate(zip(files, pool.map(cached_scan, files)), 1):
            for match in result['matches']:
                found.append(dict(match, file=entry['filename'], global_row=offset + match['row']))
            offset += result['rows']
            if number % 60 == 0:
                counts = {h: len({x['tweet_id'] for x in found if x['account'] == h}) for h in CANDIDATES}
                print('SCANNED', number, 'ROWS', offset, 'COUNTS', counts, flush=True)
    result = {'dataset': DATASET, 'revision': REVISION, 'source_revision': SOURCE_REVISION,
              'files': len(files), 'rows': offset, 'matches': found,
              'counts': {h: len({x['tweet_id'] for x in found if x['account'] == h}) for h in CANDIDATES}}
    (cache / 'discovery.json').write_text(json.dumps(result))
    print('FINAL COUNTS', result['counts'], flush=True)
    return result



def import_sample(cache, output, accounts, per_outlet=500):
    discovery = json.loads((cache / 'discovery.json').read_text())
    if len(accounts) != 5 or len(set(accounts)) != 5 or not {'nytimes', 'foxnews'} <= set(accounts):
        raise ValueError('Choose five distinct accounts including nytimes and foxnews')
    selected = []
    for account in accounts:
        if discovery['counts'].get(account, 0) <= 500:
            raise ValueError(f'{account} does not have over 500 unique tweets')
        unique = {}
        for row in discovery['matches']:
            if row['account'] == account:
                unique.setdefault(row['tweet_id'], row)
        ranked = sorted(unique.values(), key=lambda row: hashlib.sha256(
            ('media-sample-v1:' + row['tweet_id']).encode()).hexdigest())
        selected.extend(ranked[:per_outlet])
    if len(selected) != len(accounts) * per_outlet:
        raise ValueError('Insufficient unique tweets for requested sample size')
    full_cache = cache / 'full_rows'
    full_cache.mkdir(exist_ok=True)
    def fetch(match):
        path = full_cache / (match['tweet_id'] + '.json')
        if path.exists():
            response = json.loads(path.read_text())
        else:
            params = {'dataset': DATASET, 'config': 'default', 'split': 'train',
                      'offset': match['global_row'], 'length': 1}
            response = get_json('https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(params))
            path.write_text(json.dumps(response))
        if len(response['rows']) != 1 or response['rows'][0]['truncated_cells']:
            raise RuntimeError('Missing row or truncated cells; refusing incomplete raw data')
        row = response['rows'][0]['row']
        url_match = URL_PATTERN.match(row['url'])
        if not url_match or url_match[1].lower() != match['account'] or url_match[2] != match['tweet_id']:
            raise RuntimeError('Viewer row does not match pinned Parquet discovery')
        if str(row['id']) != match['tweet_id']:
            raise RuntimeError('Tweet ID mismatch')
        return response['features'], row
    raw, features = [], None
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index, (row_features, row) in enumerate(pool.map(lambda x: retry(fetch, x), selected), 1):
            if features is not None and features != row_features:
                raise RuntimeError('Inconsistent source schema')
            features = row_features
            raw.append(row)
            if index % 100 == 0: print('IMPORTED', index, '/', len(selected), flush=True)
    # Restore original scalar types, including exact 64-bit integer tweet IDs.
    schema = pa.schema([(f['name'], pa.type_for_alias(f['type']['dtype'])) for f in features])
    table = pa.Table.from_pylist(raw, schema=schema)
    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output / 'raw_tweets.parquet', compression='zstd')
    (output / 'selection.json').write_text(json.dumps(selected, indent=2) + '\n')
    (output / 'source_schema.json').write_text(json.dumps(features, indent=2) + '\n')
    manifest = {k: discovery[k] for k in ('dataset', 'revision', 'source_revision', 'files', 'rows', 'counts')}
    manifest.update({'selected_accounts': accounts, 'tweets_per_outlet': per_outlet,
                     'sample_size': len(raw), 'raw_columns': table.num_columns,
                     'sampling': 'First occurrence per URL tweet ID; lowest SHA256(media-sample-v1:<tweet_id>) per account',
                     'raw_sha256': hashlib.sha256((output / 'raw_tweets.parquet').read_bytes()).hexdigest()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('SAVED', len(raw), 'rows with', table.num_columns, 'columns', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--import-sample', action='store_true')
    parser.add_argument('--accounts', nargs=5, default=['nytimes', 'foxnews', 'reuters', 'ap', 'abc'])
    parser.add_argument('--output', type=Path, default=Path('dataset'))
    args = parser.parse_args()
    if args.import_sample:
        import_sample(args.cache, args.output, args.accounts)
    else:
        discover(args.cache)


if __name__ == '__main__': main()
