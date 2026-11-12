"""Read source changes, including the contents of changed submodule revisions."""

import os
from pathlib import Path
import re
import subprocess
import time


DOC_SUFFIXES = {'.md', '.rst', '.adoc'}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.PIPE)


def baseline(root, value='', policy=None):
    provided = bool(value and set(value) != {'0'})
    if provided:
        if not re.fullmatch(r'[0-9a-fA-F]{40,64}', value):
            raise ValueError('base revision must be a commit hash')
    else:
        value = git(root, 'rev-parse', 'HEAD^').decode().strip()
    if policy and os.environ.get('GITHUB_EVENT_NAME') == 'schedule':
        # Activating selective CI must not replay pre-policy repository history.
        additions = git(root, 'log', '--reverse', '--diff-filter=A', '--format=%H', '--', policy).decode().splitlines()
        if additions:
            start = git(root, 'rev-parse', additions[0] + '^').decode().strip()
            ensure_commit(root, value)
            older = subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', value, start],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not provided or older.returncode == 0:
                value = start
    return value


def ensure_commit(root, revision):
    if subprocess.run(['git', '-C', str(root), 'cat-file', '-e', revision + '^{commit}'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        command = ['git', '-C', str(root), '-c', 'fetch.recurseSubmodules=false',
                   'fetch', '--quiet', '--no-tags', '--depth=1', 'origin', revision]
        for attempt in range(3):
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                return
            if b'shallow file has changed since we read it' not in result.stderr or attempt == 2:
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
            # A competing shallow update may have finished while fetch exited.
            if subprocess.run(['git', '-C', str(root), 'cat-file', '-e', revision + '^{commit}'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                return
            time.sleep(0.1 * (attempt + 1))


def changed_paths(root, before, after='HEAD'):
    root = Path(root).resolve()
    ensure_commit(root, before)
    records = git(root, 'diff', '--raw', '-z', '--no-renames', '--no-abbrev', before, after).split(b'\0')
    for index in range(0, len(records) - 1, 2):
        fields = records[index].decode('ascii').split()
        old_mode, new_mode, old_sha, new_sha, _ = fields
        path = os.fsdecode(records[index + 1])
        if old_mode == ':160000' and new_mode == '160000':
            child = root / path
            if not (child / '.git').exists():
                subprocess.run(['git', '-C', str(root), 'submodule', 'update', '--init',
                                '--depth=1', '--', path], check=True)
            yield from (path + '/' + name for name in changed_paths(child, old_sha, new_sha))
        else:
            yield path


def documentation(path):
    p = Path(path)
    name = p.name.lower()
    return (p.suffix.lower() in DOC_SUFFIXES or name.startswith(('license', 'copying', 'copyright'))
            or name in {'readme', 'authors', 'contributors', '.gitignore', '.gitattributes'}
            or 'LICENSES' in p.parts or 'docs' in p.parts)


def emit(result):
    import json
    def encode(value):
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str):
            return value
        return json.dumps(value, separators=(',', ':'))
    lines = [f'{key}={encode(value)}' for key, value in result.items()]
    output = os.environ.get('GITHUB_OUTPUT')
    if output:
        with open(output, 'a', encoding='utf-8') as stream:
            stream.write('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))
