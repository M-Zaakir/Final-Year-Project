"""Freeze a Morse dataset manifest and perform reproducible OFFLINE preprocessing.

Raw CSV/JSON files are read only. Model inputs never include key_state, motor
command, target, speed, strength, condition IDs or filenames. Reference events
are saved separately; their intended labels are only pulse-count aligned.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

CORE = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + list('0123456789') + [
    'HELLO','WORLD','MORSE','SIGNAL','SENSOR','RULER','TEST','CODE','DATA','SOS',
    'HELLO WORLD','TEST CODE','A7K2M','3N8QF','B5R0X','9C4LP','D6T1Z',
    '2H7VW','E0J8S','4U9GA','F1Y6N','8M3KD','Q2B7R','5Z0TC']
PAIR_TARGETS = ['ET','SOS','TEST','CODE','A1B2','7K3M']
SPEEDS = ['slow','normal','fast']
STRENGTHS = ['low','medium','high']
SENSOR_COLUMNS = ['accel_x_raw','accel_y_raw','accel_z_raw',
                  'gyro_x_raw','gyro_y_raw','gyro_z_raw']
CONFIG = dict(version='1.0', resample_hz=833.0, baseline_seconds=0.25,
              bandpass_hz=[120.0,230.0], butterworth_order=4,
              rms_window_ms=20.0, offline_zero_phase=True,
              model_channels=['az_centered_raw','az_bandpassed_raw','az_rms_envelope_raw'],
              units='sensor raw counts; no unverified physical-unit conversion',
              source='pilot engineering findings; provisional until training/validation checks',
              scope='offline analysis only; not a causal/embedded latency benchmark')

def digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def reference_runs(a):
    if a.empty: return []
    ts=a.timestamp_us.to_numpy(np.int64); key=a.key_state.to_numpy(np.int64)
    transition=np.flatnonzero(np.diff(key))+1
    starts=np.r_[0,transition]; ends=np.r_[transition,len(a)]
    terminal=int(ts[-1]+(np.median(np.diff(ts)) if len(ts)>1 else 0))
    return [(int(key[s]),int(ts[s]),int(ts[e]) if e<len(ts) else terminal)
            for s,e in zip(starts,ends)]

def expected_slots(s):
    slots={}
    for j,target in enumerate(CORE):
        c=(j+s)%9
        slots[(target,SPEEDS[c//3],STRENGTHS[c%3],'clean')]=('C'+str(c),'')
    for p,target in enumerate(PAIR_TARGETS):
        c=(p+s)%9
        for noise in ['clean','table_tap']:
            slots[(target,SPEEDS[c//3],STRENGTHS[c%3],noise)]=(
                'C'+str(c),'P01_MAIN'+str(s+1).zfill(2)+'_MATCH_'+target)
    assert len(slots)==72
    return slots

def inspect_capture(path,d):
    a=pd.read_csv(path); n=len(a); reasons=[]
    if n<2: return a,dict(n=n,hz='',pulses=0),['empty_or_too_short']
    t=a.timestamp_us.to_numpy(np.int64); di=np.diff(a.sample_index); dt=np.diff(t)
    if (di!=1).any(): reasons.append('sample_index_discontinuity')
    if (dt<=0).any(): reasons.append('nonmonotonic_timestamps')
    if (np.abs(a[SENSOR_COLUMNS].to_numpy(np.int64))>=32760).any(): reasons.append('sensor_clipping')
    if a[SENSOR_COLUMNS].isna().any().any(): reasons.append('nonfinite_sensor_data')
    if not a.key_state.isin([0,1]).all(): reasons.append('invalid_key_state')
    if not (a.system_state==3).all(): reasons.append('unexpected_system_state')
    if not (a.status_flags==13).all(): reasons.append('unexpected_status_flags')
    expected=np.where(a.key_state.to_numpy()==1,int(d['drive_command']),0)
    if not np.array_equal(a.motor_command.to_numpy(),expected): reasons.append('motor_command_mismatch')
    if d.get('parser_crc_errors_total',0): reasons.append('logger_crc_errors')
    hz=(n-1)*1e6/(t[-1]-t[0]) if t[-1]>t[0] else 0
    if not 0.95*833<=hz<=1.05*833: reasons.append('unexpected_effective_rate')
    runs=reference_runs(a); pulses=sum(k==1 for k,_,_ in runs)
    expected_pulses=sum(x in '.-' for x in d.get('target_morse',''))
    if pulses!=expected_pulses: reasons.append('pulse_count_mismatch')
    return a,dict(n=n,hz=hz,pulses=pulses),reasons

def build_manifest(dataset,out):
    out.mkdir(parents=True,exist_ok=True); rows=[]; hashes=[]; missing=[]
    slots={s:expected_slots(s-1) for s in range(1,10)}
    for path in sorted((dataset/'recordings').glob('P01_MAIN*.json')):
        d=json.loads(path.read_text(encoding='utf-8-sig'))
        match=re.fullmatch(r'P01_MAIN(0[1-9])(B)?',d['session_id'])
        if not match: raise ValueError('Unrecognised main session: '+d['session_id'])
        s=int(match[1]); group='P01_MAIN'+str(s).zfill(2)
        combo=(d['target_text'],d['speed_condition'],d['drive_strength'],d['noise_condition'])
        condition,pair=slots[s].get(combo,('',''))
        csvpath=path.with_suffix('.csv');a,qc,reasons=inspect_capture(csvpath,d)
        notes=d.get('recording_notes','')
        if notes.strip().lower()=='mistake': reasons.append('operator_marked_mistake')
        if d['session_id']=='P01_MAIN01B': reasons.append('geometry_unconfirmed_350mm_vs_320mm')
        if not condition: reasons.append('outside_planned_target_condition_slot')
        if d.get('nominal_sample_rate_hz')!=833: reasons.append('nominal_rate_metadata_mismatch')
        row=dict(capture_id=path.stem,session_id=d['session_id'],group_session=group,
                 split='train' if s<=6 else 'validation' if s==7 else 'test',
                 target_text=d['target_text'],target_morse=d['target_morse'],
                 speed=d['speed_condition'],strength=d['drive_strength'],
                 drive_command=d['drive_command'],noise=d['noise_condition'],
                 condition_id=condition,pair_id=pair,complete_noise_pair=0,
                 include_primary=0,reason=';'.join(reasons),label_status='intended_text_pulse_count_only',
                 sample_count=qc['n'],effective_rate_hz=qc['hz'],observed_pulses=qc['pulses'],
                 csv_relative=csvpath.relative_to(dataset).as_posix(),
                 json_relative=path.relative_to(dataset).as_posix(),
                 csv_sha256=digest(csvpath),json_sha256=digest(path),original_notes=notes)
        row['_combo']=combo;row['_s']=s
        rows.append(row)
    # Prefer explicit redo/replacement notes, then a matching Cn note, then
    # chronological first capture. No decoder/model performance is used.
    for s in range(1,10):
        for combo,(condition,pair) in slots[s].items():
            candidates=[r for r in rows if r['_s']==s and r['_combo']==combo and not r['reason']]
            if candidates:
                def priority(r):
                    n=r['original_notes'].lower()
                    return (0 if ('redo' in n or 'replacement' in n) else 1,
                            0 if re.search(r'\b'+condition.lower()+r'\b',n) else 1,r['capture_id'])
                candidates.sort(key=priority);candidates[0]['include_primary']=1
                candidates[0]['reason']='selected_planned_slot'
                for r in candidates[1:]:r['reason']='duplicate_or_superseded_capture'
            else:
                missing.append(dict(group_session='P01_MAIN'+str(s).zfill(2),
                                    target_text=combo[0],speed=combo[1],strength=combo[2],
                                    noise=combo[3],condition_id=condition,
                                    action='omitted_under_authorised_no_replacement_policy'))
    groups=collections.defaultdict(list)
    for r in rows:
        if r['include_primary'] and r['pair_id']:groups[r['pair_id']].append(r)
    for rs in groups.values():
        complete=len(rs)==2 and {r['noise'] for r in rs}=={'clean','table_tap'}
        for r in rs:r['complete_noise_pair']=int(complete)
    for r in rows:
        del r['_s'];del r['_combo']
    fields=list(rows[0]);write_csv(out/'inclusion_manifest.csv',rows,fields)
    write_csv(out/'omitted_planned_slots.csv',missing,['group_session','target_text','speed','strength','noise','condition_id','action'])
    for path in sorted(dataset.rglob('*')):
        if path.is_file() and path.suffix in ('.csv','.json') and 'P01_MAIN' in path.name:
            hashes.append(dict(path=path.relative_to(dataset).as_posix(),sha256=digest(path)))
    write_csv(out/'source_hashes.csv',hashes,['path','sha256'])
    included=[r for r in rows if r['include_primary']]
    summary=dict(total_main_attempts=len(rows),included=len(included),excluded=len(rows)-len(included),
                 split_counts=dict(collections.Counter(r['split'] for r in included)),
                 session_counts=dict(collections.Counter(r['group_session'] for r in included)),
                 complete_noise_pairs=sum(r['complete_noise_pair'] for r in included)//2,
                 omitted_slots=len(missing),
                 label_limitation='Pulse count is not proof of correct dots/dashes or character/word gaps.',
                 unresolved_geometry='MAIN01B excluded conservatively: recorded 350 mm, established rig 320 mm. No silent correction.',
                 raw_files_changed=False)
    (out/'manifest_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'preprocessing_config.json').write_text(json.dumps(CONFIG,indent=2))
    print(json.dumps(summary,indent=2))

def transform(a,config=CONFIG,all_channels=False):
    # Uses only timestamps and IMU measurements; never key/motor/reference labels.
    raw_t=a.timestamp_us.to_numpy(np.float64);t=(raw_t-raw_t[0])/1e6
    if len(t)<32 or (np.diff(t)<=0).any():raise ValueError('Insufficient/nonmonotonic data')
    fs=float(config['resample_hz']);grid=np.arange(int(np.floor(t[-1]*fs))+1)/fs
    channels=SENSOR_COLUMNS if all_channels else ['accel_z_raw']
    uniform=np.column_stack([np.interp(grid,t,a[c].to_numpy(float)) for c in channels])
    base=grid<float(config['baseline_seconds'])
    if base.sum()<10:raise ValueError('Insufficient baseline interval')
    baseline=np.median(uniform[base],axis=0);centered=uniform-baseline
    z=centered[:,channels.index('accel_z_raw')]
    sos=butter(int(config['butterworth_order']),config['bandpass_hz'],btype='bandpass',fs=fs,output='sos')
    filtered=sosfiltfilt(sos,z)
    n=max(3,int(round(config['rms_window_ms']*fs/1000)))
    if n%2==0:n+=1
    padded=np.pad(filtered**2,(n//2,n//2),mode='reflect')
    envelope=np.sqrt(np.maximum(0,np.convolve(padded,np.ones(n)/n,mode='valid')))
    features=np.column_stack([centered,filtered,envelope]).astype(np.float32)
    names=([c.replace('_raw','_centered_raw') for c in channels] if all_channels
           else ['az_centered_raw'])+['az_bandpassed_raw','az_rms_envelope_raw']
    if not np.isfinite(features).all():raise ValueError('Nonfinite processed values')
    return grid,features,names,baseline

def export_reference(a,row,path):
    runs=reference_runs(a);pulse_positions=[i for i,(k,_,_)in enumerate(runs)if k==1]
    labels=[x for x in row['target_morse'] if x in '.-'];events=[];i=0;t0=int(a.timestamp_us.iloc[0])
    for state,start,end in runs:
        event_label='gap'
        if state==1:
            event_label=labels[i] if i<len(labels) else 'unmatched';i+=1
        events.append(dict(start_s=(start-t0)/1e6,end_s=(end-t0)/1e6,
                           key_reference=state,intended_symbol=event_label,
                           annotation_status='count_aligned_not_symbol_verified'))
    write_csv(path,events,['start_s','end_s','key_reference','intended_symbol','annotation_status'])

def preprocess(dataset,out,splits,limit=None,all_channels=False):
    manifest=out/'inclusion_manifest.csv'
    if not manifest.exists():raise ValueError('Run manifest first')
    with manifest.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    selected=[r for r in rows if r['include_primary']=='1'and r['split']in splits]
    if limit is not None:selected=selected[:limit]
    results=[];plot_examples={};sos_cfg=json.loads((out/'preprocessing_config.json').read_text())
    for row in selected:
        src=dataset/row['csv_relative']
        if digest(src)!=row['csv_sha256']:raise ValueError('Raw file changed: '+str(src))
        if digest(dataset/row['json_relative'])!=row['json_sha256']:raise ValueError('Metadata changed; rebuild manifest')
        a=pd.read_csv(src);grid,x,names,baseline=transform(a,sos_cfg,all_channels)
        target=out/'model_inputs'/row['split']/(row['capture_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,X=x,feature_names=np.asarray(names),sample_rate_hz=np.asarray(sos_cfg['resample_hz']),
                            source_t0_us=np.asarray(int(a.timestamp_us.iloc[0])),baseline_raw=baseline)
        ref=out/'reference_only'/row['split']/(row['capture_id']+'_events.csv')
        export_reference(a,row,ref)
        key=a.key_state.to_numpy();raw_t=(a.timestamp_us.to_numpy()-a.timestamp_us.iloc[0])/1e6
        baseline_pressed=int(key[raw_t<sos_cfg['baseline_seconds']].sum())
        # Reference diagnostics are not fed into X and never change parameters.
        z=2 if all_channels else 0
        rmse=float(np.sqrt(np.mean((x[:,z]-np.interp(grid,raw_t,a.accel_z_raw)-(-baseline[z]))**2)))
        results.append(dict(capture_id=row['capture_id'],split=row['split'],samples=len(grid),
                            uniform_rate_hz=sos_cfg['resample_hz'],input_effective_rate_hz=row['effective_rate_hz'],
                            finite=bool(np.isfinite(x).all()),baseline_key_pressed_samples=baseline_pressed,
                            max_envelope_raw=float(x[:,-1].max()),resampling_check_rmse_raw=rmse,
                            model_input_path=target.relative_to(out).as_posix(),
                            reference_path=ref.relative_to(out).as_posix()))
        cell=(row['speed'],row['strength'])
        if row['split']=='train'and(cell not in plot_examples or row['noise']=='table_tap'and plot_examples[cell][0]['noise']=='clean'):
            plot_examples[cell]=(row,grid,x,a)
    if results:
        name='preprocessing_qc_'+'_'.join(splits)+'.csv'
        write_csv(out/name,results,list(results[0]))
    if plot_examples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,3,figsize=(15,10),sharey=False)
        for i,speed in enumerate(SPEEDS):
            for j,strength in enumerate(STRENGTHS):
                ax=axes[i,j];item=plot_examples.get((speed,strength))
                if item is None:ax.set_visible(False);continue
                row,g,x,a=item;ax.plot(g,x[:,-1],color='#165c8a',lw=.8,label='Vibration RMS envelope')
                key=a.key_state.to_numpy();tt=(a.timestamp_us.to_numpy()-a.timestamp_us.iloc[0])/1e6
                ax.fill_between(tt,0,float(x[:,-1].max()),where=key==1,color='#c88527',alpha=.16,label='Button reference only')
                ax.set_title(f"{speed}, {strength}: {row['target_text']} ({row['noise']})",fontsize=9)
                ax.set_xlabel('Time (s)');ax.set_ylabel('Envelope (raw counts)');ax.grid(alpha=.15)
        fig.suptitle('Training-only preprocessing check: vibration envelope and separate button reference',fontsize=13)
        handles,labels=next(ax for ax in axes.flat if ax.get_visible()).get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=2);fig.tight_layout(rect=[0,.04,1,.96])
        (out/'figures').mkdir(exist_ok=True);fig.savefig(out/'figures/training_preprocessing_overview.png',dpi=160);plt.close(fig)
    print(f'Processed {len(results)} included recordings. Splits: {splits}. No scaling/model fitting performed.')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('MorseDataset'))
    parser.add_argument('--output',type=Path,default=Path('MorsePrepared'))
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('manifest')
    p=sub.add_parser('preprocess');p.add_argument('--splits',nargs='+',choices=['train','validation','test'],default=['train'])
    p.add_argument('--limit',type=int);p.add_argument('--all-channels',action='store_true')
    args=parser.parse_args()
    if args.command=='manifest':build_manifest(args.dataset,args.output)
    else:preprocess(args.dataset,args.output,args.splits,args.limit,args.all_channels)

if __name__=='__main__':main()
