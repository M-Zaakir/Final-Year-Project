import json,unittest,tempfile
from pathlib import Path
import error_analysis as ea

class DiagnosticTests(unittest.TestCase):
    def test_minimum_edits_preserve_spaces_and_unknowns(self):
        for a,b in [('HELLO WORLD','HELLOWORLD'),('0','5'),('TEST','NST'),('SOS','?'),('','A'),('A','')]:
            self.assertEqual(sum(ea.text_edits(a,b).values()),ea.base.edit_distance(a,b))
        self.assertEqual(ea.text_edits('HELLO WORLD','HELLOWORLD')['deletions'],1)

    def test_combined_flags_are_explicit(self):
        self.assertEqual(ea.category(1,0,2,3),'detection+pulse_label+gap_label')
        self.assertEqual(ea.category(0,0,0,0),'none')

    def test_completed_diagnostics_reconcile(self):
        p=Path('MorseErrors_Local')
        if not (p/'error_summary.csv').exists():p=Path(__file__).parent/'MorseErrors'
        if not (p/'error_summary.csv').exists():self.skipTest('Run analysis first')
        rows=ea.base.read_csv(p/'message_diagnostics.csv');summaries=ea.base.read_csv(p/'error_summary.csv');cats=ea.base.read_csv(p/'failure_categories.csv')
        self.assertEqual(len(rows),720)
        for s in summaries:
            r=[x for x in rows if x['method']==s['method']];self.assertEqual(len(r),144)
            self.assertEqual(sum(int(x['character_edits'])for x in r),int(s['character_edits']))
            self.assertEqual(sum(int(x['substitutions'])+int(x['deletions'])+int(x['insertions'])for x in r),int(s['character_edits']))
            self.assertEqual(sum(int(x['failed_messages'])for x in cats if x['method']==s['method']),int(s['failed_messages']))
        self.assertFalse(any(r['diagnostic_category']=='none' and r['exact_message']=='0'for r in rows))

    def test_input_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d);(p/'frozen_protocol.json').write_text(json.dumps(dict(train_capture_ids=['train'],test_capture_ids=['test'])))
            (p/'test_predictions.csv').write_text('original')
            (p/'complete.json').write_text(json.dumps(dict(protocol_sha256=ea.base.sha(p/'frozen_protocol.json'),files_sha256={'test_predictions.csv':ea.base.sha(p/'test_predictions.csv')})))
            ea.validate(p)
            (p/'test_predictions.csv').write_text('altered')
            with self.assertRaisesRegex(ValueError,'Output hash mismatch'):ea.validate(p)

if __name__=='__main__':unittest.main()
