import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('security_check', Path(__file__).with_name('check.py'))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
POLICY = json.loads(Path(__file__).with_name('minimum-versions.json').read_text())


class SecurityChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = copy.deepcopy(POLICY)
        self.policy['legacy_backports'] = []

    def write(self, path, text):
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def findings(self):
        return check.inspect(self.root, self.policy)[0]

    def test_manifest_alias_and_workspace_floor(self):
        self.write('Cargo.toml', '[workspace.dependencies]\nwebpki={package="rustls-webpki",version="0.103.13"}\n')
        self.assertEqual(self.findings()[0]['package'], 'rustls-webpki')

    def test_patched_and_future_versions_pass(self):
        self.write('Cargo.lock', '[[package]]\nname="rustls-webpki"\nversion="0.103.16"\n[[package]]\nname="webpki-roots"\nversion="1.0.11"\n')
        self.assertEqual(self.findings(), [])

    def test_old_lock_and_unsupported_node_are_blocked(self):
        self.write('Cargo.lock', '[[package]]\nname="rustls-webpki"\nversion="0.103.15"\n')
        self.write('.node-version', '24.19.0\n')
        self.assertEqual({x['package'] for x in self.findings()}, {'rustls-webpki','node'})

    def test_unbounded_python_dependency_is_blocked(self):
        self.write('pyproject.toml', '[project]\nname="demo"\nversion="1"\ndependencies=["certifi"]\n')
        self.assertEqual(self.findings()[0]['package'], 'certifi')

    def test_constraints_and_generated_caches(self):
        self.write('pyproject.toml', '[project]\nname="demo"\nversion="1"\ndependencies=["certifi>=2026.10.19"]\n')
        self.write('target/Cargo.lock', 'broken!')
        self.assertEqual(self.findings(), [])

    def test_legacy_version_without_patch_fails(self):
        self.write('Cargo.lock', '[[package]]\nname="rustls-webpki"\nversion="0.102.8"\n')
        self.assertTrue(self.findings())

    def test_workspace_members_use_the_workspace_lock(self):
        self.write('Cargo.toml','[workspace]\nmembers=["member"]\n')
        self.write('Cargo.lock','[[package]]\nname="rustls-webpki"\nversion="0.103.16"\n')
        self.write('member/Cargo.toml','[package]\nname="member"\nversion="1.0.0"\n')
        self.write('member/Cargo.lock','[[package]]\nname="rustls-webpki"\nversion="0.103.13"\n')
        self.assertEqual(self.findings(),[])
        # If exported as a standalone project, the same stale lock is blocked.
        self.assertTrue(check.inspect(self.root/'member',self.policy)[0])

    def legacy(self):
        self.write('Cargo.toml', '[patch.crates-io]\npatched={package="rustls-webpki",path="vendor/patched"}\n')
        self.write('Cargo.lock', '[[package]]\nname="rustls-webpki"\nversion="0.102.8"\n[[package]]\nname="webpki-roots"\nversion="0.25.4"\n')
        self.write('vendor/patched/Cargo.toml', '[package]\nname="rustls-webpki"\nversion="0.102.8"\n')
        self.write('vendor/patched/src/lib.rs', '// synthetic reviewed backport fixture\n')
        self.policy['legacy_backports'] = [{'path':'vendor/patched','manifest':'Cargo.toml','version':'0.102.8','sha256':check.tree_hash(self.root/'vendor/patched')}]

    def test_reviewed_backport_and_legacy_roots_pass(self):
        self.legacy()
        self.assertEqual(self.findings(), [])

    def test_backport_tampering_or_removed_patch_fails(self):
        self.legacy()
        self.write('vendor/patched/src/lib.rs', '// protection removed\n')
        self.assertTrue(self.findings())
        self.write('Cargo.toml', '[package]\nname="demo"\nversion="1"\n')
        self.assertTrue(self.findings())

    def test_private_registry_is_not_queried(self):
        self.write('package-lock.json', json.dumps({'packages':{'node_modules/private':{'version':'1.0.0','resolved':'https://private.example/pkg.tgz'},'node_modules/public':{'version':'2.0.0','resolved':'https://registry.npmjs.org/public/-/public-2.0.0.tgz'}}}))
        _, packages = check.inspect(self.root, self.policy)
        self.assertEqual(list(packages), [('npm','public','2.0.0')])

    def test_unsupported_locks_do_not_claim_advisory_coverage(self):
        self.write('pnpm-lock.yaml', 'lockfileVersion: 9.0\n')
        self.write('node_modules/example/yarn.lock', 'generated')
        result = check.coverage(self.root, {})
        self.assertEqual(result['advisory_coverage'], 'none')
        self.assertEqual(result['unsupported_lockfiles'], ['pnpm-lock.yaml'])

    def test_advisory_pagination_and_network_failure(self):
        class Response:
            def __init__(self, data):self.data=data
            def __enter__(self):return self
            def __exit__(self, *args):pass
            def read(self):return json.dumps(self.data).encode()
        packages={('crates.io','example','1.0.0'):{'files':['Cargo.lock'],'backported':False}}
        responses=[Response({'results':[{'vulns':[{'id':'CVE-A'}],'next_page_token':'next'}]}),Response({'results':[{'vulns':[{'id':'CVE-B'}]}]})]
        with patch.object(check.urllib.request,'urlopen',side_effect=responses) as request:
            found=check.query_osv(packages)
            self.assertEqual([v['advisory'] for v in found], ['CVE-A','CVE-B'])
            body=json.loads(request.call_args_list[1].args[0].data)
            self.assertEqual(body['queries'][0]['page_token'],'next')
        with patch.object(check.urllib.request,'urlopen',side_effect=OSError('unavailable')):
            with self.assertRaises(OSError):check.query_osv(packages)

    def test_backport_exempts_only_the_reviewed_incident(self):
        class Response:
            def __init__(self,data):self.data=data
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return json.dumps(self.data).encode()
        packages={('crates.io','rustls-webpki','0.102.8'):{'files':['Cargo.lock'],'backported':True}}
        responses=[Response({'results':[{'vulns':[{'id':'INCIDENT'},{'id':'OTHER'}]}]}),
                   Response({'aliases':['CVE-2026-111943']}),Response({'aliases':['CVE-OTHER']})]
        with patch.object(check.urllib.request,'urlopen',side_effect=responses):
            findings=check.query_osv(packages)
        self.assertEqual([v['advisory'] for v in findings],['OTHER'])


if __name__ == '__main__':
    unittest.main()
