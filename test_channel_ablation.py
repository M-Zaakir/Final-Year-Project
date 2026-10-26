import inspect
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from channel_ablation import channel_stats,feature_columns,feature_names,check_normaliser,CHANNEL_SETS

class ChannelFeatureTests(unittest.TestCase):
    def test_no_annotation_inputs(self):
        self.assertEqual(list(inspect.signature(channel_stats).parameters),['band','noise','intervals','fs'])

    def test_channel_selection_is_nested_and_correct(self):
        for kind,n in [('pulse',13),('gap',8)]:
            envelope=feature_columns(kind,'envelope_only');z=feature_columns(kind,'az')
            xyz=feature_columns(kind,'accel_xyz');six=feature_columns(kind,'accel_gyro')
            self.assertEqual(envelope,list(range(n)))
            self.assertEqual(z,list(range(n))+list(range(n+10,n+15)))
            self.assertTrue(set(envelope)<set(z)<set(xyz)<set(six))
            self.assertEqual(len(feature_names(kind)),n+30)

    def test_sensor_permutation_moves_only_corresponding_features(self):
        t=np.arange(1000)/1000
        signals=np.column_stack([(i+1)*np.sin(2*np.pi*174*t)for i in range(6)])
        intervals=np.array([[.2,.5],[.6,.8]])
        a=channel_stats(signals,np.ones(6),intervals,1000).reshape(2,6,5)
        permutation=[2,1,0,5,4,3]
        b=channel_stats(signals[:,permutation],np.ones(6),intervals,1000).reshape(2,6,5)
        np.testing.assert_allclose(b,a[:,permutation],atol=1e-10)

    def test_empty_short_and_quiet_segments_are_finite(self):
        x=channel_stats(np.zeros((100,6)),np.zeros(6),np.array([[.02,.021],[.03,.02]]),1000)
        self.assertEqual(x.shape,(2,30));self.assertTrue(np.isfinite(x).all())
        self.assertEqual(channel_stats(np.zeros((100,6)),np.ones(6),np.empty((0,2)),1000).shape,(0,30))

    def test_normaliser_scope_rejects_held_out_fit(self):
        train=[{'row':{'capture_id':'train'}}];held=[{'row':{'capture_id':'held'}}]
        check_normaliser({'training_capture_ids':['train']},train,held)
        with self.assertRaises(AssertionError):check_normaliser({'training_capture_ids':['train','held']},train,held)

class CompletedRunTests(unittest.TestCase):
    def test_all_configurations_share_captures_and_test_is_reserved(self):
        parent=Path(__file__).parent
        root=parent/'MorseChannels_Local'if(parent/'MorseChannels_Local').exists()else parent/'MorseChannels'
        protocol=json.loads((root/'run_protocol.json').read_text())
        self.assertFalse(protocol['test_loaded']);self.assertTrue(protocol['raw_unchanged'])
        d=pd.read_csv(root/'channel_comparison.csv')
        self.assertEqual(len(d),24)
        self.assertEqual(set(d.channel_set),set(CHANNEL_SETS))
        self.assertTrue(d.loc[d.scope=='MAIN07_validation','recordings'].eq(71).all())
        self.assertTrue(d.loc[d.scope=='grouped_training_cv','recordings'].eq(423).all())
        folds=json.loads((root/'fold_provenance.json').read_text())
        for f in folds:self.assertFalse(set(f['fit_ids'])&set(f['heldout_ids']))
        for name in ['validation_predictions.csv','out_of_fold_predictions.csv']:
            p=pd.read_csv(root/name)
            self.assertFalse(p.group_session.isin(['P01_MAIN08','P01_MAIN09']).any())
            counts=p.groupby(['model','channel_set']).capture_id.nunique()
            self.assertEqual(len(counts),12)
            self.assertTrue(counts.eq(71 if name.startswith('validation')else 423).all())

if __name__=='__main__':unittest.main()
