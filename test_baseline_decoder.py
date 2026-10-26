import inspect
import json
from pathlib import Path
import unittest
import numpy as np
from baseline_decoder import predict,match_events,classify_timing,edit_distance,MORSE

def example_model():
    return dict(detection=dict(fraction=.2,close_gap_s=.015,min_pulse_s=.045),
                onset_bias_s=0,release_bias_s=0,
                timing=dict(log_unit_mean=np.log(.1),log_unit_sigma=.35,
                    pulse=dict(means=np.log([1,3]).tolist(),sigmas=[.25,.25],priors=[.5,.5]),
                    gap=dict(means=np.log([1,3,7]).tolist(),sigmas=[.25]*3,priors=[.65,.30,.05])))

def envelope_for(codes,unit=.1,fs=1000):
    intervals=[];t=1.
    for i,token in enumerate(codes):
        if token=='/':t+=4*unit;continue
        for j,symbol in enumerate(token):
            end=t+unit*(1 if symbol=='.'else 3);intervals.append((t,end));t=end
            if j<len(token)-1:t+=unit
        if i<len(codes)-1:t+=3*unit
    env=np.zeros(int((t+1)*fs))
    for a,b in intervals:env[round(a*fs):round(b*fs)]=100
    return env,fs,np.array(intervals)

class DecoderTests(unittest.TestCase):
    def test_sos_decodes_from_envelope(self):
        env,fs,ref=envelope_for(['...','---','...'])
        result=predict(env,fs,example_model())
        self.assertEqual(result['text'],'SOS')
        self.assertEqual(len(match_events(result['intervals'],ref)),9)

    def test_word_gap_is_preserved(self):
        env,fs,_=envelope_for(['.','/','-'])
        self.assertEqual(predict(env,fs,example_model())['text'],'E T')

    def test_amplitude_change_does_not_change_text(self):
        env,fs,_=envelope_for(['.-','-...'])
        self.assertEqual(predict(env,fs,example_model())['text'],
                         predict(env*2.7,fs,example_model())['text'])

    def test_quiet_input_gives_no_message(self):
        self.assertEqual(predict(np.zeros(2000),1000,example_model())['text'],'')

    def test_short_impulse_is_rejected(self):
        env=np.zeros(2000);env[1000:1010]=100
        self.assertEqual(predict(env,1000,example_model())['text'],'')

    def test_predict_api_has_no_label_or_condition_inputs(self):
        self.assertEqual(list(inspect.signature(predict).parameters),['envelope','fs','model'])
        env,fs,_=envelope_for(['...','---','...']);model=example_model()
        a=predict(env,fs,model)['text']
        model.update(target_text='WRONG',speed='wrong',strength='wrong',key_state='wrong')
        self.assertEqual(a,predict(env,fs,model)['text'])

    def test_matching_cannot_reuse_reference(self):
        detected=np.array([[1,1.1],[1.02,1.12]])
        self.assertEqual(len(match_events(detected,np.array([[1,1.1]]))),1)

    def test_character_edit_distance_and_alphabet(self):
        self.assertEqual(edit_distance('HELLO WORLD','HELLOWORLD'),1)
        self.assertEqual(edit_distance('ABC','AXC'),1)
        self.assertEqual(len(MORSE),36)

if __name__=='__main__':unittest.main()
