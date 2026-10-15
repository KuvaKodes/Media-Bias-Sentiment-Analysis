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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    args = parser.parse_args()
    discover(args.cache)


if __name__ == '__main__': main()
