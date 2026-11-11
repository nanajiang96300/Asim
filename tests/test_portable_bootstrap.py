import hashlib,importlib.util,io,json,tarfile,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('asim_bootstrap',Path(__file__).resolve().parents[1]/'scripts/bootstrap.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
class BootstrapTests(unittest.TestCase):
    def fixture(self,r,bad=None):
        cache=r/'cache';cache.mkdir();raw=b'known source bytes';manifest=json.dumps({'files':[{'path':'vendor/a.h','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}]}).encode()
        p=cache/'cpp.tar.gz'
        with tarfile.open(p,'w:gz') as t:
            for name,data in [('vendor/a.h',raw),('DEPENDENCY_MANIFEST.json',manifest)]:
                i=tarfile.TarInfo(name);i.size=len(data);t.addfile(i,io.BytesIO(data))
            if bad:
                i=tarfile.TarInfo('../escape' if bad=='traversal' else 'vendor/link');i.type=tarfile.REGTYPE if bad=='traversal' else tarfile.SYMTYPE;i.linkname='/etc/passwd';t.addfile(i,io.BytesIO())
        spec={'dependency_commit':'example','source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),'artifacts':[{'file':p.name,'bytes':p.stat().st_size,'sha256':b.digest(p),'kind':'cpp-source'}]}
        return cache,spec,p
    def test_valid_offline_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);cache,spec,_=self.fixture(r);out=r/'installed';self.assertEqual(b.prepare(out,spec,'lock',cache,True),1);self.assertEqual(b.verify(out,spec,'lock'),1)
            (out/'vendor/a.h').write_text('corrupted')
            with self.assertRaises(ValueError):b.prepare(out,spec,'lock',cache,True)
    def test_bad_cached_sha_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);cache,spec,p=self.fixture(r);p.write_bytes(b'bad')
            with self.assertRaises(ValueError):b.prepare(r/'installed',spec,'lock',cache,True)
            self.assertEqual(p.read_bytes(),b'bad');self.assertFalse((r/'installed').exists())
    def test_archive_traversal_and_symlinks_rejected(self):
        for bad in ['traversal','symlink']:
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp);cache,spec,_=self.fixture(r,bad)
                with self.assertRaises(ValueError):b.prepare(r/'installed',spec,'lock',cache,True)
                self.assertFalse((r/'installed').exists());self.assertFalse((r/'escape').exists())
    def test_missing_offline_and_wrong_lock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);cache,spec,p=self.fixture(r);b.prepare(r/'installed',spec,'lock',cache,True)
            with self.assertRaises(ValueError):b.verify(r/'installed',spec,'other-lock')
            p.unlink()
            with self.assertRaises(ValueError):b.prepare(r/'other',spec,'lock',cache,True)
if __name__=='__main__':unittest.main()
