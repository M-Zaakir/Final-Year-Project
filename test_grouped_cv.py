import csv
import json
from pathlib import Path
import unittest
from grouped_cv import fold_rows,assert_no_leakage,GROUPS,METHODS

class SplitTests(unittest.TestCase):
    def test_each_training_capture_held_out_once(self):
        rows=[dict(capture_id=g,group_session=g,include_primary='1',split='train')for g in GROUPS]
        rows += [dict(capture_id='validation',group_session='P01_MAIN07',include_primary='1',split='validation'),
                 dict(capture_id='test',group_session='P01_MAIN08',include_primary='1',split='test'),
                 dict(capture_id='excluded',group_session='P01_MAIN01',include_primary='0',split='train')]
        held=[]
        for g in GROUPS:
            fold=fold_rows(rows,g)
            self.assertEqual(len(fold),6)
            held.extend(r['capture_id']for r in fold if r['split']=='validation')
            self.assertEqual({r['group_session']for r in fold if r['split']=='train'},set(GROUPS)-{g})
        self.assertEqual(sorted(held),sorted(GROUPS))
        self.assertEqual(rows[0]['split'],'train')

    def test_outer_validation_or_test_cannot_be_requested(self):
        with self.assertRaises(ValueError):fold_rows([],'P01_MAIN07')
        with self.assertRaises(ValueError):fold_rows([],'P01_MAIN09')

    def test_overlap_is_detected(self):
        with self.assertRaises(AssertionError):assert_no_leakage(['a','b'],['b','c'])
        assert_no_leakage(['a'],['b'])

class CompletedRunTests(unittest.TestCase):
    def test_all_learned_stages_have_fold_scoped_provenance(self):
        parent=Path(__file__).parent
        root=parent/'MorseCV_Local'if(parent/'MorseCV_Local').exists()else parent/'MorseCV'
        seen=set()
        for held in GROUPS:
            p=json.loads((root/'folds'/held/'fold_provenance.json').read_text())
            self.assertTrue(p['fit_scope_checks_passed'])
            self.assertTrue(p['baseline_refitted']and p['timing_normaliser_refitted'])
            self.assertTrue(p['event_features_rebuilt']and p['scalers_train_fold_only']and p['classifiers_refitted'])
            self.assertEqual(set(p['fit_sessions']),set(GROUPS)-{held})
            self.assertFalse(set(p['training_capture_ids'])&set(p['heldout_capture_ids']))
            self.assertFalse(seen&set(p['heldout_capture_ids']));seen.update(p['heldout_capture_ids'])
        self.assertEqual(len(seen),423)
        with(root/'out_of_fold_predictions.csv').open(newline='',encoding='utf-8')as f:rows=list(csv.DictReader(f))
        for method in METHODS:
            ids=[r['capture_id']for r in rows if r['model']==method]
            self.assertEqual(len(ids),423);self.assertEqual(set(ids),seen)
        s=json.loads((root/'grouped_cv_summary.json').read_text())
        self.assertFalse(s['heldout_outer_validation_loaded']);self.assertFalse(s['final_test_loaded'])

if __name__=='__main__':unittest.main()
