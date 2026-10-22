#!/usr/bin/env python3
"""Read manifests/locks without installing or executing project dependencies.

Minimum versions prevent known regressions. --advisories also queries OSV for
new disclosures; network errors fail the check. Only public registry package
identities/versions are queried. This command never updates, merges or deploys.
"""
import argparse
import hashlib
import fnmatch
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib
import urllib.request

EXCLUDE = {'.git', '.claude', '.cargo', '.cargo-home', '.rustup', '.cache',
           'node_modules', 'target', '.venv', 'venv', '__pycache__', 'artifacts'}
NAMES = {'Cargo.toml', 'Cargo.lock', 'package-lock.json', 'poetry.lock', 'uv.lock', 'pyproject.toml',
         'requirements.txt', '.node-version', '.nvmrc', 'go.sum'}
UNSUPPORTED_LOCKS = {'pnpm-lock.yaml', 'yarn.lock', 'Package.resolved', 'Pipfile.lock',
                     'bun.lock', 'bun.lockb', 'Gemfile.lock', 'composer.lock'}


def coverage(root, packages):
    """State coverage explicitly; a policy pass is not an all-ecosystem audit."""
    unsupported = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE)
        unsupported.extend(str(Path(parent, name).relative_to(root))
                           for name in sorted(set(files) & UNSUPPORTED_LOCKS))
    return {'public_package_versions': len(packages),
            'advisory_coverage': 'none' if not packages else ('partial' if unsupported else 'supported-lockfiles'),
            'unsupported_lockfiles': unsupported,
            'limits': 'Only resolved public registry dependencies in supported lockfiles are queried. '
                      'Private/git dependencies, vendored forks and installed binaries require separate review.'}


def version(value):
    match = re.fullmatch(r'v?([0-9]+(?:\.[0-9]+){0,3})', value.strip())
    if not match:
        raise ValueError('not a stable numeric version')
    nums = tuple(map(int, match[1].split('.')))
    return nums + (0,) * (3 - len(nums))


def lower_bound(requirement):
    match = re.fullmatch(r'\s*(?:\^|~|=|>=)?\s*([0-9]+(?:\.[0-9]+){0,2})\s*', requirement)
    if match:
        return version(match[1])
    # Cargo comma-separated comparator ranges are intersections.
    if ',' in requirement:
        bounds = [lower_bound(x) for x in requirement.split(',')
                  if not x.strip().startswith(('<', '*'))]
        return max(bounds) if bounds else (0, 0, 0)
    return (0, 0, 0)


def tree_hash(directory):
    digest = hashlib.sha256()
    if (directory / 'src').is_symlink():
        raise ValueError('backport source must not be symlinked')
    files = [directory / 'Cargo.toml'] + sorted((directory / 'src').rglob('*'))
    for file in files:
        if not file.is_file():
            continue
        if file.is_symlink() or not file.resolve().is_relative_to(directory.resolve()):
            raise ValueError('backport source must not be symlinked')
        digest.update(str(file.relative_to(directory)).encode() + bytes([0]))
        digest.update(file.read_bytes() + bytes([0]))
    return digest.hexdigest()


def discover(root):
    # Include local untracked manifests/backports too. Ignore generated trees.
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE)
        for name in sorted(set(files) & NAMES):
            path = Path(parent, name)
            if not path.is_symlink():
                yield path


def dependency_tables(data):
    for key, item in data.items():
        if key in {'dependencies', 'dev-dependencies', 'build-dependencies'} and isinstance(item, dict):
            yield item
        elif isinstance(item, dict):
            yield from dependency_tables(item)


def ignored_member_lock(path, root):
    """Cargo uses the enclosing workspace lock, never a member's old copy.

    A copied standalone member is checked again when scanned as its own root.
    Nested/excluded workspaces retain their independent lockfiles.
    """
    member = path.parent
    own = member / 'Cargo.toml'
    if own.exists() and 'workspace' in tomllib.loads(own.read_text()):
        return False
    for parent in member.parents:
        if not parent.is_relative_to(root):
            break
        manifest = parent / 'Cargo.toml'
        if not manifest.exists():
            continue
        workspace = tomllib.loads(manifest.read_text()).get('workspace')
        if workspace is None:
            continue
        relative = str(member.relative_to(parent))
        if any(fnmatch.fnmatchcase(relative, p) for p in workspace.get('exclude', [])):
            return False
        return (parent / 'Cargo.lock').is_file() and any(
            fnmatch.fnmatchcase(relative, p) for p in workspace.get('members', []))
    return False


def inspect(root, policy):
    root = root.resolve()
    findings = []
    packages = {}
    backports = []

    def issue(path, package, detail):
        findings.append({'file':str(path.relative_to(root)), 'package':package, 'detail':detail})

    for record in policy.get('legacy_backports', []):
        try:
            directory = (root / record['path']).resolve()
            manifest = (root / record['manifest']).resolve()
            directory.relative_to(root)
            manifest.relative_to(root)
            spec = tomllib.loads((directory / 'Cargo.toml').read_text())['package']
            patches = tomllib.loads(manifest.read_text()).get('patch', {}).get('crates-io', {})
            referenced = any(isinstance(p, dict)
                and p.get('package', alias) == spec['name']
                and (manifest.parent / p.get('path', '__missing__')).resolve() == directory
                for alias, p in patches.items())
            if spec['name'] not in {'rustls-webpki','webpki'} or spec['version'] != record['version'] or not referenced:
                raise ValueError('backport manifest/patch reference differs')
            if tree_hash(directory) != record['sha256']:
                raise ValueError('backport source fingerprint differs')
            backports.append((manifest.parent, spec['version']))
        except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as error:
            issue(root / record.get('path', '.security'), 'legacy-backport', str(error))

    def patched(path, item):
        return item['name'] in {'rustls-webpki','webpki'} and not item.get('source') and any(
            v == item['version'] and path.parent.is_relative_to(parent)
            for parent, v in backports)

    def add(ecosystem, name, ver, path, backport=False):
        item = packages.setdefault((ecosystem, name, ver), {'files':[], 'backported':True})
        item['files'].append(str(path.relative_to(root)))
        item['backported'] &= backport

    for path in discover(root):
        try:
            if path.name == 'Cargo.toml':
                data = tomllib.loads(path.read_text())
                for table in dependency_tables(data):
                    for alias, spec in table.items():
                        name = spec.get('package', alias) if isinstance(spec, dict) else alias
                        req = spec.get('version') if isinstance(spec, dict) else spec
                        if name == 'webpki' and not any(path.parent.is_relative_to(parent)
                                                       and version(v)[:2] == (0,22)
                                                       for parent, v in backports):
                            issue(path, name, 'retired verifier needs an approved local backport')
                        if name not in policy['cargo'] or not isinstance(req, str):
                            continue
                        lower = lower_bound(req)
                        if name == 'rustls-webpki' and lower[:2] in {(0, 101), (0, 102)}:
                            if not any(path.parent.is_relative_to(parent) and version(v)[:2] == lower[:2]
                                       for parent, v in backports):
                                issue(path, name, 'legacy declaration needs an approved local backport')
                            continue
                        if name.startswith('webpki-root') and lower[0] == 0:
                            if lower[:2] != (0, 26) or lower >= (0, 26, 11):
                                continue
                            issue(path, name, 'compatibility bridge minimum must be 0.26.11')
                        elif lower < version(policy['cargo'][name]):
                            issue(path, name, 'declared minimum is below '+policy['cargo'][name])
            elif path.name == 'Cargo.lock':
                if ignored_member_lock(path, root):
                    continue
                items = tomllib.loads(path.read_text()).get('package', [])
                verifiers = [p for p in items if p['name'] in {'rustls-webpki', 'webpki'}]
                protected = bool(verifiers) and all(patched(path, p) or (
                    p['name'] == 'rustls-webpki' and version(p['version']) >= (0, 103, 16)) for p in verifiers)
                for item in items:
                    name, ver = item['name'], item['version']
                    backport = patched(path, item)
                    if name == 'webpki' and not backport:
                        issue(path, name, 'retired verifier lacks a verified backport')
                    if name in policy['cargo']:
                        current = version(ver)
                        allowed_legacy_roots = name.startswith('webpki-root') and current[0] == 0 and protected
                        if current < version(policy['cargo'][name]) and not backport and not allowed_legacy_roots:
                            issue(path, name, ver+' is below the safe minimum without a verified backport')
                    if item.get('source') == 'registry+https://github.com/rust-lang/crates.io-index':
                        add('crates.io', name, ver, path, backport)
                    elif backport:
                        add('crates.io', name, ver, path, True)
            elif path.name == 'package-lock.json':
                data = json.loads(path.read_text())
                for location, item in data.get('packages', {}).items():
                    if not location or item.get('link') or not item.get('version'):
                        continue
                    if not item.get('resolved', '').startswith('https://registry.npmjs.org/'):
                        continue
                    name = item.get('name') or location.rsplit('node_modules/', 1)[-1]
                    add('npm', name, item['version'], path)
            elif path.name == 'pyproject.toml':
                data = tomllib.loads(path.read_text())
                dependencies = list(data.get('project', {}).get('dependencies', []))
                for group in data.get('project', {}).get('optional-dependencies', {}).values():
                    dependencies.extend(group)
                for name, spec in data.get('tool', {}).get('poetry', {}).get('dependencies', {}).items():
                    dependencies.append(name + (spec.get('version', '') if isinstance(spec, dict) else spec))
                for spec in dependencies:
                    match = re.match(r'^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?(.*)', spec)
                    if match and match[1].lower() in policy['python']:
                        name = match[1].lower()
                        if lower_bound(match[2].split(';')[0]) < version(policy['python'][name]):
                            issue(path, name, 'declared minimum is below '+policy['python'][name])
            elif path.name in {'uv.lock', 'poetry.lock'}:
                for item in tomllib.loads(path.read_text()).get('package', []):
                    source = item.get('source', {})
                    registry = source.get('registry') or source.get('url')
                    if registry and registry.rstrip('/') not in {'https://pypi.org/simple','https://pypi.org'}:
                        continue
                    name, ver = item['name'], item.get('version')
                    if not ver:
                        continue
                    if name in policy['python'] and version(ver) < version(policy['python'][name]):
                        issue(path, name, ver+' is below '+policy['python'][name])
                    add('PyPI', name, ver, path)
            elif path.name == 'requirements.txt':
                for line in path.read_text().splitlines():
                    match = re.match(r'^([A-Za-z0-9_.-]+)==([0-9][A-Za-z0-9.+-]*)\s*(?:;|#|--|$)',line.strip())
                    if match:
                        name, ver = match.groups()
                        if name.lower() in policy['python'] and version(ver) < version(policy['python'][name.lower()]):
                            issue(path, name, ver+' is below '+policy['python'][name.lower()])
                        add('PyPI', name, ver, path)
            elif path.name in {'.nvmrc','.node-version'}:
                text = path.read_text().strip()
                try:
                    current = version(text)
                    floor = policy['node'].get(str(current[0]))
                    if not floor or current < version(floor):
                        issue(path, 'node', 'runtime pin is outside the patched supported branches')
                except ValueError:
                    issue(path, 'node', 'runtime pin needs an explicit patched version')
            elif path.name == 'go.sum':
                for line in path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 3 and not parts[1].endswith('/go.mod'):
                        add('Go', parts[0], parts[1], path)
        except (ValueError, KeyError, TypeError, OSError, tomllib.TOMLDecodeError) as error:
            issue(path, 'parser', type(error).__name__)
    return findings, packages


def query_osv(packages):
    """Query fresh public advisories, following result pagination."""
    findings = []
    identities = list(packages)
    for offset in range(0, len(identities), 500):
        batch = identities[offset:offset+500]
        queries = [{'package':{'ecosystem':e,'name':n},'version':v} for e,n,v in batch]
        pending = list(range(len(batch)))
        while pending:
            request = urllib.request.Request('https://api.osv.dev/v1/querybatch',
                data=json.dumps({'queries':[queries[i] for i in pending]}).encode(),
                headers={'content-type':'application/json','user-agent':'project-security-check/1'})
            with urllib.request.urlopen(request, timeout=60) as response:
                results = json.load(response)['results']
            if len(results) != len(pending):
                raise ValueError('OSV returned an incomplete batch')
            next_pending = []
            for i, result in zip(pending, results):
                identity = batch[i]
                for vuln in result.get('vulns', []):
                    if packages[identity]['backported']:
                        # Exempt only this incident after source/patch verification.
                        with urllib.request.urlopen('https://api.osv.dev/v1/vulns/'+vuln['id'], timeout=30) as response:
                            detail = json.load(response)
                        if 'CVE-2026-111943' in detail.get('aliases', []):
                            continue
                    findings.append({'package':identity[1],'version':identity[2],
                        'advisory':vuln['id'],'files':packages[identity]['files']})
                if result.get('next_page_token'):
                    queries[i]['page_token'] = result['next_page_token']
                    next_pending.append(i)
            pending = next_pending
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--advisories', action='store_true')
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        policy = json.loads((args.policy or root/'.security/minimum-versions.json').read_text())
        findings, packages = inspect(root, policy)
        if args.advisories:
            findings += query_osv(packages)
        result = {'status':'failed' if findings else 'passed','findings':findings,
                  'public_package_versions_checked':len(packages),'fresh_advisories':args.advisories,
                  'coverage':coverage(root, packages)}
    except Exception as error:
        # Do not print arbitrary exception strings: registry/proxy URLs may be private.
        result = {'status':'error','error':type(error).__name__,'findings':[]}
    print(json.dumps(result, indent=2))
    return 2 if result['status']=='error' else int(result['status']!='passed')


if __name__ == '__main__':
    sys.exit(main())
