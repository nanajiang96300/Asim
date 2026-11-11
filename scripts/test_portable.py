"""Run the original numerical suite; explicitly skip optional CDL generation when absent."""
from pathlib import Path
import importlib.util,os,sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));os.chdir(ROOT)
suite=unittest.defaultTestLoader.discover(str(ROOT/'verification/tests'))
def cases(group):
    for test in group:
        if isinstance(test,unittest.TestSuite):yield from cases(test)
        else:yield test
if not all(importlib.util.find_spec(n) for n in ['torch','sionna']):
    for test in cases(suite):
        if test.id()=='test_channel_study.ChannelStudyTests.test_real_cdl_reproducible_and_not_kronecker_surrogate':
            setattr(test,test._testMethodName,unittest.skip('optional Torch/Sionna CDL generator not installed')(lambda:None))
raise SystemExit(0 if unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful() else 1)
