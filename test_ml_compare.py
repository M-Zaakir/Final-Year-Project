import inspect
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import joblib
from ml_compare import event_features,predict_ml,decode_symbols,PULSE_FEATURES,GAP_FEATURES

class FeatureTests(unittest.TestCase):
    def test_feature_api_excludes_annotations(self):
        self.assertEqual(list(inspect.signature(event_features).parameters),['envelope','fs','intervals','unit'])
        self.assertEqual(list(inspect.signature(predict_ml).parameters),['envelope','fs','baseline_model','classifiers'])
        forbidden=['target','key_state','motor','strength','speed','session','capture_id']
        self.assertFalse(any(word in name for word in forbidden for name in PULSE_FEATURES+GAP_FEATURES))

    def test_amplitude_normalisation_and_finite_shapes(self):
        env=np.zeros(2000);env[400:500]=100;env[700:1000]=100
        intervals=np.array([[.4,.5],[.7,1.]])
        p,g=event_features(env,1000,intervals,.1)
        pp,gg=event_features(env*3,1000,intervals,.1)
        np.testing.assert_allclose(p,pp,atol=1e-12)
        np.testing.assert_allclose(g,gg,atol=1e-12)
        self.assertEqual(p.shape,(2,13));self.assertEqual(g.shape,(1,8))
        self.assertTrue(np.isfinite(p).all()and np.isfinite(g).all())

    def test_no_events_and_single_event(self):
        p,g=event_features(np.ones(1000),1000,np.empty((0,2)),.1)
        self.assertEqual(p.shape,(0,13));self.assertEqual(g.shape,(0,8))
        p,g=event_features(np.ones(1000),1000,np.array([[.2,.3]]),.1)
        self.assertEqual(p.shape,(1,13));self.assertEqual(g.shape,(0,8))

    def test_symbol_gap_conversion_and_unknown_tokens(self):
        self.assertEqual(decode_symbols(['.','-'],['character']),'ET')
        self.assertEqual(decode_symbols(['.','-'],['word']),'E T')
        self.assertEqual(decode_symbols(['.','-'],['element']),'A')
        self.assertEqual(decode_symbols(['.']*6,['element']*5),'?')

class FitProvenanceTests(unittest.TestCase):
    def test_train_only_provenance_and_scaler_statistics(self):
        root=Path(__file__).parent/'MorseML'
        f=pd.read_csv(root/'training_event_features.csv')
        self.assertTrue(f.session.str.fullmatch(r'P01_MAIN0[1-6]').all())
        for name in ['logistic_regression','rbf_svm','random_forest']:
            bundle=joblib.load(root/'models'/(name+'.joblib'))
            self.assertEqual(bundle['fit_split'],'train')
            self.assertEqual(len(bundle['training_capture_ids']),423)
            self.assertFalse(any('P01_MAIN07'in s or'P01_MAIN08'in s or'P01_MAIN09'in s for s in bundle['training_capture_ids']))
            if name!='random_forest':
                for kind,fields in [('pulse',PULSE_FEATURES),('gap',GAP_FEATURES)]:
                    expected=f.loc[f.event_type==kind,fields].to_numpy(float).mean(axis=0)
                    np.testing.assert_allclose(bundle[kind].named_steps['standardscaler'].mean_,expected,atol=1e-10)

if __name__=='__main__':unittest.main()
