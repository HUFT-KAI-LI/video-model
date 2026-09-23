"""Export complete-run compact evidence; never put model weights or full videos in git."""
import argparse
import json
import shutil
from pathlib import Path
from protocol import HERE, MODELS, jobs, digest, write_json, write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    a=p.parse_args();run=a.run.resolve();out=HERE/'evidence'
    expected={j['video_id'] for m in MODELS for j in jobs(m)}
    manifests=[json.loads(p.read_text()) for p in sorted(run.glob('outputs/*/complete.json'))]
    if {x['job']['video_id'] for x in manifests}!=expected:raise ValueError('Need every planned output')
    validation=json.loads((run/'evaluation/validation.json').read_text())
    videos = validation['videos']
    if (validation['completed'] != len(expected) or validation['expected'] != len(expected)
            or len(videos) != len(expected)
            or {x['video_id'] for x in videos} != expected
            or any(not x['pts_verified'] or x['frames'] != 80 for x in videos)):
        raise ValueError('Need complete frame validation')
    summary=json.loads((run/'evaluation/summary.json').read_text())
    if summary['completed_videos']!=160:raise ValueError('Need complete scores')
    out.mkdir(exist_ok=True)
    for name in ['plan.json','weights_verified.json']:
        shutil.copyfile(run/name,out/name)
    for name in ['proxy_scores.csv','summary.json','validation.json','provenance.json']:
        shutil.copyfile(run/'evaluation'/name,out/name)
    shutil.copyfile(run/'evaluation/RESULTS.md',HERE/'RESULTS.md')
    write_json(out/'generation_manifests.json',manifests)
    write_json(out/'worker_loads.json',[json.loads(p.read_text()) for p in sorted(run.glob('*_load.json'))])
    write_json(out/'interrupted_attempts.json',[json.loads(p.read_text()) for p in sorted(run.glob('interrupted_attempts/*/interrupted.json'))])
    statuses=json.loads((out/'RUN_STATUS.json').read_text())
    statuses.update(status='160 videos generated, frame/hash validation and automatic proxy complete; human review pending',
                    generated=160,validated=160,human_accuracy=summary['human'])
    write_json(out/'RUN_STATUS.json',statuses)
    # Inspectable frame evidence for every sample, independent of the inferred score.
    previews=out/'initial_frames';previews.mkdir(exist_ok=True)
    for vid in sorted(expected):
        shutil.copyfile(run/'evaluation'/(vid+'_initial.jpg'),previews/(vid+'.jpg'))
    write_json(out/'artifact_checksums.json',{str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
                                             if p.is_file() and p.name!='artifact_checksums.json'})
    print('Exported complete-run evidence; human labels remain pending.')


if __name__=='__main__':main()
