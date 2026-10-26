"""Leakage-controlled, leave-one-training-session-out comparison.

Only MAIN01--06 are used. Every learned detector/timing/scaler/classifier
stage is fitted again within each fold. Fixed pilot-derived waveform
preprocessing is reused. MAIN07--09 are not loaded by this runner.
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import sys

import joblib
import numpy as np
import sklearn
import scipy
import baseline_decoder as base
import ml_compare as ml

GROUPS=['P01_MAIN'+str(i).zfill(2)for i in range(1,7)]
METHODS=['duration_baseline']+ml.MODEL_NAMES

def read_rows(path):
    with Path(path).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))

def fold_rows(rows,heldout):
    if heldout not in GROUPS:raise ValueError('Only MAIN01--06 may be held out')
    outer=[dict(r)for r in rows if str(r['include_primary'])=='1'and r['split']=='train']
    if set(r['group_session']for r in outer)!=set(GROUPS):raise ValueError('Six training groups required')
    for r in outer:
        if r['group_session']not in GROUPS:raise ValueError('Unexpected outer training group')
        r['split']='validation'if r['group_session']==heldout else'train'
    return outer

def assert_no_leakage(fit_ids,held_ids):
    if set(fit_ids)&set(held_ids):raise AssertionError('Held-out capture used for fitting')

def fingerprint(prepared):
    rows=read_rows(prepared/'inclusion_manifest.csv')
    selected=[r for r in rows if r['include_primary']=='1'and r['split']=='train']
    index={r['capture_id']:r for r in read_rows(prepared/'preprocessing_qc_train.csv')}
    inputs=[]
    for r in selected:
        e=index[r['capture_id']]
        inputs.append(dict(capture_id=r['capture_id'],npz_sha256=base.sha(prepared/e['model_input_path']),
                           reference_sha256=base.sha(prepared/e['reference_path'])))
    info=dict(manifest_sha256=base.sha(prepared/'inclusion_manifest.csv'),
              preprocessing_sha256=base.sha(prepared/'preprocessing_config.json'),
              runner_sha256=base.sha(__file__),baseline_code_sha256=base.sha(base.__file__),
              ml_code_sha256=base.sha(ml.__file__),numpy=np.__version__,sklearn=sklearn.__version__,
              joblib=joblib.__version__,scipy=scipy.__version__,python=sys.version,
              training_input_hashes=inputs)
    key=hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest()
    return key,info

def make_fold_prepared(prepared,destination,heldout):
    rows=fold_rows(read_rows(prepared/'inclusion_manifest.csv'),heldout)
    destination.mkdir(parents=True,exist_ok=True)
    base.write_csv(destination/'inclusion_manifest.csv',rows,list(rows[0]))
    shutil.copy2(prepared/'preprocessing_config.json',destination/'preprocessing_config.json')
    source={r['capture_id']:r for r in read_rows(prepared/'preprocessing_qc_train.csv')}
    indexes={'train':[],'validation':[]}
    for row in rows:
        item=dict(source[row['capture_id']]);item['split']=row['split']
        # Absolute pointers are temporary only; raw/preprocessed inputs are not copied.
        item['model_input_path']=str((prepared/item['model_input_path']).resolve())
        item['reference_path']=str((prepared/item['reference_path']).resolve())
        indexes[row['split']].append(item)
    for split,items in indexes.items():
        base.write_csv(destination/('preprocessing_qc_'+split+'.csv'),items,list(items[0]))
    return rows

def validate_fit(prepared,fold_baseline,fold_ml,heldout,fold_records):
    train=[r['capture_id']for r in fold_records if r['split']=='train']
    held=[r['capture_id']for r in fold_records if r['split']=='validation']
    b=json.loads((fold_baseline/'baseline_model.json').read_text())
    if set(b['training_capture_ids'])!=set(train):raise AssertionError('Baseline fit scope mismatch')
    assert_no_leakage(b['training_capture_ids'],held)
    features=read_rows(fold_ml/'training_event_features.csv')
    if set(r['session']for r in features)!=set(GROUPS)-{heldout}:raise AssertionError('Feature-label group mismatch')
    for name in ml.MODEL_NAMES:
        bundle=joblib.load(fold_ml/'models'/(name+'.joblib'))
        if set(bundle['training_capture_ids'])!=set(train):raise AssertionError('ML fit scope mismatch')
        assert_no_leakage(bundle['training_capture_ids'],held)
        if name!='random_forest':
            for kind,columns in [('pulse',ml.PULSE_FEATURES),('gap',ml.GAP_FEATURES)]:
                values=np.array([[float(r[c])for c in columns]for r in features if r['event_type']==kind])
                np.testing.assert_allclose(bundle[kind].named_steps['standardscaler'].mean_,values.mean(axis=0),atol=1e-10)
    return dict(heldout_session=heldout,fit_sessions=[g for g in GROUPS if g!=heldout],
                train_recordings=len(train),heldout_recordings=len(held),training_capture_ids=train,
                heldout_capture_ids=held,baseline_refitted=True,timing_normaliser_refitted=True,
                event_features_rebuilt=True,scalers_train_fold_only=True,classifiers_refitted=True,
                fit_scope_checks_passed=True)

def run(prepared,out):
    prepared=prepared.resolve();out.mkdir(parents=True,exist_ok=True)
    key,inputs=fingerprint(prepared)
    info_path=out/'run_inputs.json'
    if info_path.exists()and json.loads(info_path.read_text())['fingerprint']!=key:
        raise ValueError('Code/data/environment changed; use a NEW output directory to preserve the previous run')
    base.dump(info_path,dict(fingerprint=key,**inputs))
    for i,heldout in enumerate(GROUPS,1):
        dest=out/'folds'/heldout
        if (dest/'complete.json').exists():
            if json.loads((dest/'complete.json').read_text())['fingerprint']!=key:raise ValueError('Fold fingerprint changed')
            print(f'Fold {i}/6: {heldout} already complete; reused.',flush=True);continue
        dest.mkdir(parents=True,exist_ok=True)
        print(f'Fold {i}/6: hold out {heldout}; refitting five-session pipeline...',flush=True)
        with tempfile.TemporaryDirectory(prefix='morse_fold_')as temporary:
            work=Path(temporary);fp=work/'prepared';fb=work/'baseline';fm=work/'ml'
            fp.mkdir();fb.mkdir();fm.mkdir()
            records=make_fold_prepared(prepared,fp,heldout)
            with (dest/'fit_and_evaluation.log').open('w',encoding='utf-8')as log,contextlib.redirect_stdout(log):
                base.fit(fp,fb)
                ml.fit(fp,fb,fm)
                scope=validate_fit(fp,fb,fm,heldout,records)
                ml.evaluate(fp,fb,fm,'validation')
            scope['fingerprint']=key
            base.dump(dest/'fold_provenance.json',scope)
            shutil.copy2(fb/'baseline_model.json',dest/'fold_baseline_model.json')
            shutil.copy2(fb/'detection_training_search.csv',dest/'detection_training_search.csv')
            for name in ['fit_summary.csv','training_provenance.json']:
                shutil.copy2(fm/name,dest/name)
            for p in fm.glob('validation_*'):
                # The existing ML plot names MAIN07; omit it for inner folds.
                # The pooled CV figure below labels the held-out sessions correctly.
                if p.is_file()and p.suffix!='.png':shutil.copy2(p,dest/p.name)
            base.dump(dest/'complete.json',dict(fingerprint=key,heldout_session=heldout,complete=True))
        scores=read_rows(dest/'validation_comparison.csv')
        print('  '+', '.join(r['model']+' CER='+f"{100*float(r['character_error_rate']):.1f}%"for r in scores),flush=True)
    summarize(out)

def summarize(out):
    allrows=[];fold_summaries=[];known_ids=None
    for held in GROUPS:
        folder=out/'folds'/held
        if not(folder/'complete.json').exists():raise ValueError('Incomplete fold '+held)
        for name in METHODS:
            rows=read_rows(folder/('validation_'+name+'_predictions.csv'))
            for r in rows:
                r['outer_heldout_session']=held
                for field in ['exact_message','character_edits','reference_characters','event_tp','event_fp','event_fn',
                              'intended_symbol_agreements','matched_reference_pulses']:r[field]=int(r[field])
                r['decode_ms']=float(r['decode_ms'])
            allrows.extend(rows)
            s=base.summarize(rows);s.update(model=name,heldout_session=held)
            fold_summaries.append(s)
    combined=[]
    for name in METHODS:
        rows=[r for r in allrows if r['model']==name]
        ids=[r['capture_id']for r in rows]
        if len(ids)!=len(set(ids)):raise AssertionError('Capture evaluated more than once')
        if known_ids is None:known_ids=set(ids)
        if set(ids)!=known_ids:raise AssertionError('Methods evaluated different capture sets')
        s=base.summarize(rows);s['model']=name
        folds=[r for r in fold_summaries if r['model']==name]
        s['mean_session_cer']=float(np.mean([r['character_error_rate']for r in folds]))
        s['session_cer_sd']=float(np.std([r['character_error_rate']for r in folds],ddof=1))
        s['min_session_cer']=min(r['character_error_rate']for r in folds)
        s['max_session_cer']=max(r['character_error_rate']for r in folds)
        combined.append(s)
    base.write_csv(out/'grouped_cv_summary.csv',combined,list(combined[0]))
    base.write_csv(out/'grouped_cv_by_session.csv',fold_summaries,list(fold_summaries[0]))
    base.write_csv(out/'out_of_fold_predictions.csv',allrows,list(allrows[0]))
    ranking=sorted(combined,key=lambda r:(r['character_error_rate'],-r['exact_message_accuracy'],r['model']))
    base.dump(out/'grouped_cv_summary.json',dict(protocol='Leave one of MAIN01--06 out; six folds',
        results=combined,provisional_training_cv_leader=ranking[0]['model'],
        ranking_rule='Pooled out-of-fold CER, then exact-message recovery, then name',
        capture_count_per_method=len(known_ids),heldout_outer_validation_loaded=False,final_test_loaded=False,
        caveat='Same operator/day/rig and recurring vocabulary. Intended labels remain count-aligned, not independently verified. No hyperparameter tuning.'))
    plot(out,fold_summaries)
    print(json.dumps(combined,indent=2),flush=True)

def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5))
    for name in METHODS:
        rs=sorted([r for r in rows if r['model']==name],key=lambda r:r['heldout_session'])
        ax.plot(range(1,7),[100*r['character_error_rate']for r in rs],marker='o',label=name.replace('_',' '))
    ax.set_xticks(range(1,7),['MAIN01','MAIN02','MAIN03','MAIN04','MAIN05','MAIN06'])
    ax.set_xlabel('Held-out training session');ax.set_ylabel('Character error rate, including spaces (%)')
    ax.set_title('Six-fold session-held-out comparison: every learned stage refitted within each fold')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'grouped_cv_by_session.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('MorsePrepared'))
    p.add_argument('--output',type=Path,default=Path('MorseCV'))
    p.add_argument('command',choices=['run','summarize'])
    a=p.parse_args()
    if a.command=='run':run(a.prepared,a.output)
    else:summarize(a.output)

if __name__=='__main__':main()
