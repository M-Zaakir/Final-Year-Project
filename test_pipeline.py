"""Meaningful signal/annotation separation and manifest integrity checks.
Run: python test_pipeline.py
"""
import csv
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from morse_pipeline import transform,CONFIG

class SignalTests(unittest.TestCase):
    def test_labels_and_commands_cannot_change_features(self):
        t=np.arange(0,3,1/824.0)
        t[1:-1]+=np.sin(np.arange(1,len(t)-1))*.00003
        a=pd.DataFrame({'timestamp_us':np.rint(t*1e6).astype(np.int64),
                        'accel_z_raw':8000+500*np.sin(2*np.pi*174*t),
                        'key_state':0,'motor_command':0})
        g,x,n,b=transform(a)
        changed=a.copy();changed['key_state']=1;changed['motor_command']=192
        gg,xx,nn,bb=transform(changed)
        np.testing.assert_array_equal(x,xx)
        np.testing.assert_array_equal(g,gg)
        self.assertEqual(n,CONFIG['model_channels'])
        self.assertTrue(np.isfinite(x).all())
        self.assertLess(abs(np.median(x[g<.25,0])),1)

    def test_bandpass_retains_carrier_and_rejects_low_frequency(self):
        t=np.arange(0,3,1/833.0)
        def response(f):
            a=pd.DataFrame({'timestamp_us':np.rint(t*1e6).astype(np.int64),
                            'accel_z_raw':8000+1000*np.sin(2*np.pi*f*t)})
            g,x,_,_=transform(a)
            mid=(g>.5)&(g<2.5)
            return np.sqrt(np.mean(x[mid,1]**2))
        self.assertGreater(response(174),500)
        self.assertLess(response(40)/response(174),.05)

    def test_time_reversal_is_rejected(self):
        a=pd.DataFrame({'timestamp_us':[0,1]*40,'accel_z_raw':np.zeros(80)})
        with self.assertRaises(ValueError):transform(a)

class ManifestTests(unittest.TestCase):
    def test_session_isolation_pair_matching_and_model_schema(self):
        root=Path(__file__).parent/'MorsePrepared'
        with (root/'inclusion_manifest.csv').open(newline='',encoding='utf-8')as f:
            rows=list(csv.DictReader(f))
        included=[r for r in rows if r['include_primary']=='1']
        seen={};pairs={}
        for r in included:
            self.assertEqual(seen.setdefault(r['group_session'],r['split']),r['split'])
            n=int(r['group_session'][-2:]);self.assertEqual(r['split'],'train'if n<=6 else'validation'if n==7 else'test')
            self.assertNotEqual(r['session_id'],'P01_MAIN01B')
            if r['complete_noise_pair']=='1':pairs.setdefault(r['pair_id'],[]).append(r)
        for members in pairs.values():
            self.assertEqual(len(members),2)
            self.assertEqual({r['noise']for r in members},{'clean','table_tap'})
            for field in ['target_text','speed','strength','split']:
                self.assertEqual(members[0][field],members[1][field])
        for p in (root/'model_inputs/train').glob('*.npz'):
            with np.load(p,allow_pickle=False)as a:
                self.assertEqual(a['X'].shape[1],3)
                self.assertEqual(a['feature_names'].tolist(),CONFIG['model_channels'])
                self.assertTrue(np.isfinite(a['X']).all())
                self.assertNotIn('key_state',a.files)
                self.assertNotIn('motor_command',a.files)
        self.assertFalse((root/'model_inputs/test').exists())
        # Phase 2 explicitly allows MAIN07 preprocessing; test remains reserved.

if __name__=='__main__':unittest.main()
