import copy,unittest
from scripts.remaining_contracts import case_contract,validate_case,validate_checks,normalize_probe

class RemainingContractTests(unittest.TestCase):
    def test_wrong_algorithm_profile_seed_or_shape_rejected(self):
        name='cholesky_block_f16_42';spec=case_contract(name);validate_case(name,spec)
        for key,value in [('operator','ldl_block_explicit_v1'),('precision','structure_f64_real_block'),('seed',123)]:
            bad=copy.deepcopy(spec);bad[key]=value
            with self.assertRaises(ValueError):validate_case(name,bad)
        bad=copy.deepcopy(spec);bad['task']['batch_size']=48
        with self.assertRaises(ValueError):validate_case(name,bad)
    def test_empty_or_missing_ldl_checks_rejected(self):
        for checks in ({},{'L':True},{'L':True,'D':True,'unit_diagonal':True,'positive_D':1}):
            with self.assertRaises(ValueError):validate_checks('ldl_block_f64',checks)
        validate_checks('ldl_block_f64',dict(L=True,D=True,unit_diagonal=True,positive_D=True))
    def test_probe_normalization_only_allows_scalar_width(self):
        a={'header':{'declared_compute_size':1,'schema':'probe'},'instructions':[{'id':'probe.scalar','compute_size':1,'opcode':19},{'id':'load','size':1}]};b=copy.deepcopy(a);b['header']['declared_compute_size']=16;b['instructions'][0]['compute_size']=16
        self.assertEqual(normalize_probe(a),normalize_probe(b));b['instructions'][1]['size']=2;self.assertNotEqual(normalize_probe(a),normalize_probe(b))
    def test_exact_direct_case_suffixes(self):
        self.assertEqual(case_contract('cholesky_merge_shape2')['task']['batch_size'],48)
        self.assertEqual(case_contract('ns_corrected_v1_regression')['algorithm']['iterations'],16)
        with self.assertRaises(ValueError):case_contract('cholesky_merge_unknown')

if __name__=='__main__':unittest.main()
