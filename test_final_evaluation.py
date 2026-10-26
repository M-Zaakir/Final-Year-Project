import inspect,json,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import final_evaluation as final

class FinalTests(unittest.TestCase):
    def test_inference_has_no_reference_inputs(self):
        self.assertEqual(list(inspect.signature(final.infer).parameters),['sensor','config','norm','bundles','z'])
        with self.assertRaises(ValueError):final.infer(pd.DataFrame({'key_state':[1]}),{},None,None,None)

    def test_scoring_keeps_spaces_unknowns_and_all_errors(self):
        row=dict(capture_id='a',group_session='P01_MAIN08',target_text='A B',speed='slow',strength='low',noise='clean',condition_id='C0',pair_id='',complete_noise_pair='0')
        pred=dict(text='A?B',intervals=np.empty((0,2)))
        s=final.score(row,pred,np.empty((0,2)))
        self.assertEqual(s['character_edits'],1);self.assertEqual(s['reference_characters'],3);self.assertEqual(s['exact_message'],0)
        self.assertEqual(final.summary([s])['character_error_rate'],1/3)

    def test_changed_frozen_artifact_is_rejected_before_data_read(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'frozen_protocol.json').write_text(json.dumps({'artifact_hashes':{'manifest':'wrong'}}))
            (p/'inclusion_manifest.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen artifact changed'):
                final.evaluate(p,p,p,p,p,p)

    def test_completed_run_scope(self):
        p=Path('MorseFinal_Local')
        if not (p/'complete.json').exists():p=Path(__file__).parent/'MorseFinal'
        if not (p/'complete.json').exists():self.skipTest('Run freeze and evaluate first')
        protocol=json.loads((p/'frozen_protocol.json').read_text());complete=json.loads((p/'complete.json').read_text())
        self.assertEqual(final.base.sha(p/'frozen_protocol.json'),complete['protocol_sha256'])
        self.assertFalse(set(protocol['train_capture_ids'])&set(protocol['test_capture_ids']))
        rows=final.base.read_csv(p/'test_predictions.csv');self.assertEqual(len(rows),144*5)
        self.assertEqual({r['group_session'] for r in rows},set(final.TEST_SESSIONS))
        for method in final.METHODS:self.assertEqual({r['capture_id'] for r in rows if r['method']==method},set(protocol['test_capture_ids']))
        for name,digest in complete['files_sha256'].items():self.assertEqual(final.base.sha(p/name),digest)

if __name__=='__main__':unittest.main()
