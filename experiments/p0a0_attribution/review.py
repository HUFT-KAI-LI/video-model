"""Build anonymous paired human review; preserve labels in browser and export CSV."""
import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from protocol import DESIGN, jobs, write_json, write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    a=p.parse_args();run=a.run.resolve();out=run/'review';media=out/'media'
    media.mkdir(parents=True,exist_ok=True)
    rng=random.Random(DESIGN['blind_order_seed']);pairs=jobs('wan');rng.shuffle(pairs)
    items=[];unblind={};blank=[]
    for index,j in enumerate(pairs):
        models=['wan','longlive'];rng.shuffle(models)
        pair=[]
        for side,model in zip('AB',models):
            video_id=model+'_'+j['pair_id'];blind_id=f'R{index+1:03d}_{side}'
            unblind[blind_id]=video_id
            folder=run/'outputs'/video_id
            if not (folder/'complete.json').exists():continue
            for src,dst in [(folder/'video.mp4',media/(blind_id+'.mp4')),
                            (run/'evaluation'/(video_id+'_initial.jpg'),media/(blind_id+'.jpg'))]:
                if src.exists() and not dst.exists(): shutil.copyfile(src,dst)
            pair.append(dict(blind_id=blind_id,side=side,video='media/'+blind_id+'.mp4',image='media/'+blind_id+'.jpg'))
            blank.append(dict(blind_id=blind_id,initial_attribute='',initial_failure_type='',later_change='',notes=''))
        if pair:items.append(dict(pair_number=index+1,prompt=j['prompt'],target=j['target_attribute'],clips=pair))
    write_json(out/'unblinding.json',unblind)
    # This is a generated blank template, never a file containing reviewer annotations.
    write_csv(out/'labels_template.csv',blank,fields=['blind_id','initial_attribute','initial_failure_type','later_change','notes'])
    data=json.dumps(items,ensure_ascii=False).replace('<','\\u003c')
    fingerprint=hashlib.sha256((run/'plan.json').read_bytes()).hexdigest()[:16]
    html=HTML.replace('__DATA__',data).replace('__KEY__',fingerprint)
    (out/'index.html').write_text(html)
    print(f'Built {len(items)} anonymous pairs at {out / "index.html"}')


HTML='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P0-A0 初始属性盲审</title>
<style>body{font:16px system-ui;margin:24px;background:#f5f6f8;color:#182330}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2}button,select,textarea{font:inherit;margin:6px;padding:7px}article{display:grid;grid-template-columns:1fr 1fr;gap:20px}section{background:white;padding:15px;border-radius:8px}video,img{width:100%}textarea{width:90%}small{display:block}#status{color:#345}p{max-width:1100px}</style>
<header><b>P0-A0 初始属性盲审</b> <button onclick="move(-1)">上一组</button><button onclick="move(1)">下一组</button><button onclick="download()">导出标注 CSV</button><span id="status"></span></header>
<p>主指标只看开头 1 秒：指定主体/物件存在且颜色正确。明确错误或遗漏选 INCORRECT；遮挡、太小或无法判定选 UNJUDGEABLE。下方缩略图依次为 0、0.25、0.5、0.75 秒。完整视频用于检查后续变化，不能据此修改初始判定。自动评分与模型名称在此隐藏。标注自动保存在当前浏览器，请导出备份。</p>
<h3 id="target"></h3><p id="prompt"></p><article id="clips"></article>
<script>
const items=__DATA__, key='p0a0_review___KEY__';let index=0;
let labels={};try{labels=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function save(id,field,value){labels[id]??={};labels[id][field]=value;localStorage.setItem(key,JSON.stringify(labels));status()}
function status(){document.getElementById('status').textContent=`组 ${index+1}/${items.length} · 已标 ${Object.values(labels).filter(x=>x.initial_attribute).length}/${items.reduce((n,x)=>n+x.clips.length,0)}`}
function select(id,field,options){const el=document.createElement('select');for(const v of options){const o=document.createElement('option');o.value=v;o.textContent=v||'待标注';el.append(o)}el.value=labels[id]?.[field]||'';el.onchange=()=>save(id,field,el.value);return el}
function render(){if(!items.length)return;const item=items[index];document.getElementById('target').textContent='目标属性：'+item.target;document.getElementById('prompt').textContent=item.prompt;const root=document.getElementById('clips');root.replaceChildren();for(const clip of item.clips){const s=document.createElement('section');const h=document.createElement('h3');h.textContent=clip.blind_id;s.append(h);const im=document.createElement('img');im.src=clip.image;im.alt='初始一秒四帧';s.append(im);const v=document.createElement('video');v.src=clip.video;v.controls=true;v.preload='metadata';s.append(v);const b=document.createElement('button');b.textContent='仅播放初始 1 秒';b.onclick=()=>{v.currentTime=0;v.ontimeupdate=()=>{if(v.currentTime>=1){v.pause();v.ontimeupdate=null}};v.play()};s.append(b);const full=document.createElement('button');full.textContent='播放完整 5 秒';full.onclick=()=>{v.ontimeupdate=null;v.currentTime=0;v.play()};s.append(full);s.append(document.createElement('br'));s.append('初始属性：',select(clip.blind_id,'initial_attribute',['','CORRECT','INCORRECT','UNJUDGEABLE']));s.append(document.createElement('br'));s.append('初始失败类型：',select(clip.blind_id,'initial_failure_type',['','none','attribute_mismatch','object_omission','identity_structure_mismatch','uncertain']));s.append(document.createElement('br'));s.append('后续是否变化：',select(clip.blind_id,'later_change',['','YES','NO','UNJUDGEABLE']));const note=document.createElement('textarea');note.placeholder='可选备注';note.value=labels[clip.blind_id]?.notes||'';note.oninput=()=>save(clip.blind_id,'notes',note.value);s.append(note);root.append(s)}status()}
function move(delta){index=Math.max(0,Math.min(items.length-1,index+delta));render()}
function download(){const fields=['blind_id','initial_attribute','initial_failure_type','later_change','notes'];const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';const rows=[fields,...items.flatMap(x=>x.clips.map(c=>fields.map(f=>f==='blind_id'?c.blind_id:labels[c.blind_id]?.[f]||'')))];const blob=new Blob(['\\ufeff'+rows.map(r=>r.map(quote).join(',')).join('\\r\\n')],{type:'text/csv;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='p0a0_human_labels.csv';a.click();URL.revokeObjectURL(a.href)}
render();</script></html>'''

if __name__=='__main__':main()
