"""Offline review UI. Explicit CSV export; never fabricates human labels."""
import html
import json
from common import REVIEW_FIELDS, TAXONOMY, read_csv


def build(run, plan, summaries):
    videos = {r['video_id']: r for r in plan['videos']}
    existing = {r['video_id']: r for r in read_csv(run / 'human_review.csv')}
    records = [dict(**videos[r['video_id']], mining=r, review=existing[r['video_id']])
               for r in sorted(summaries, key=lambda r: ({'candidate':0, 'audit':1, 'unselected':2}[r['selection']], r['video_id']))]
    payload = json.dumps(records).replace('<', '\\u003c')
    fields = json.dumps(REVIEW_FIELDS)
    options = ''.join(f'<option>{html.escape(t)}</option>' for t in TAXONOMY)
    page = '''<!doctype html><meta charset="utf-8"><title>P0-A 人工复核</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#f6f7fa;color:#172338}video{width:100%;max-height:520px;background:#111}label{display:inline-block;margin:10px}input,select,button{font:inherit;padding:7px}textarea{width:95%;height:60px}button{cursor:pointer}#preview{max-width:100%}.box{background:white;padding:18px;border-radius:12px;margin:14px 0}</style>
<h1>P0-A 自然失败人工复核</h1><p>先看早期参考，再看完整视频。候选分数不是标签。转身、暗光、遮挡、出画本身不等于失败。所有索引从 0 开始。</p>
<p>YES：确认失败；NO：已看到 reviewed_until_sec 且未见失败；UNCERTAIN：无法确定。gradual_drift 无法定位首次错误时留空 onset。pre_onset_normal 只由人工判断。</p>
<button id="prev">上一条</button> <select id="choose"></select> <button id="next">下一条</button> <button id="save">导出 human_review.csv</button>
<div class="box"><h2 id="title"></h2><p id="prompt"></p><p id="proposal"></p><video id="video" controls preload="metadata"></video><p id="position"></p>
<button id="onset">以当前帧标记 onset</button><button id="watched">以当前时间标记已复核范围</button></div>
<div class="box" id="form">
<label>是否失败 <select data-key="failure_confirmed"><option></option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select></label>
<label>错误类型 <select data-key="failure_type"><option></option>OPTIONS</select></label>
<label>onset block <input data-key="failure_onset_block" type="number" min="0"></label>
<label>onset frame（可选） <input data-key="failure_onset_frame" type="number" min="0"></label>
<label>起始形态 <select data-key="onset_kind"><option></option><option>abrupt</option><option>gradual_drift</option></select></label>
<label>此前正常 <select data-key="pre_onset_normal"><option></option><option>YES</option><option>NO</option><option>UNCERTAIN</option></select></label>
<label>已复核至秒 <input data-key="reviewed_until_sec" type="number" min="0" step="0.0625"></label>
<label>置信度 <select data-key="human_confidence"><option></option><option>high</option><option>medium</option><option>low</option></select></label>
<textarea data-key="notes" placeholder="说明早期不符合 prompt、参考不稳定、遮挡、无法定位等情况"></textarea>
</div><details open><summary>自动曲线（不代表人工标签）</summary><img id="curve" style="max-width:100%" onerror="this.hidden=true" onload="this.hidden=false"></details><details><summary>每个 block 的预览（不替代完整视频）</summary><img id="preview"></details>
<p>更换视频时保存在当前页面内存；关闭前必须导出 CSV，替换本次 run 的 human_review.csv。未导出的更改会丢失。</p>
<script>
const records=PAYLOAD, fields=FIELDS, blockMap=BLOCKMAP;
let index=0; const $=s=>document.querySelector(s), inputs=[...document.querySelectorAll('[data-key]')];
function persist(){for(const e of inputs)records[index].review[e.dataset.key]=e.value;}
function show(){const r=records[index]; $('#choose').value=index; $('#title').textContent=r.video_id+' / '+r.mining.selection; $('#prompt').textContent=r.prompt; $('#proposal').textContent='自动建议 block: '+r.mining.suggested_onset_block+'（必须人工核验）'; $('#video').src='outputs/'+r.video_id+'/video.mp4'; $('#preview').src='outputs/'+r.video_id+'/preview.jpg'; $('#curve').src='outputs/'+r.video_id+'/scores.png';for(const e of inputs)e.value=r.review[e.dataset.key]||'';}
records.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.video_id+' / '+r.mining.selection;$('#choose').append(o)});
$('#choose').onchange=()=>{persist();index=Number($('#choose').value);show()};
$('#prev').onclick=()=>{persist();index=Math.max(0,index-1);show()};$('#next').onclick=()=>{persist();index=Math.min(records.length-1,index+1);show()};
function frame(){return Math.min(records[index].num_frames-1,Math.floor($('#video').currentTime*records[index].fps));}
function block(){return blockMap.find(b=>b.frame_start<=frame()&&frame()<b.frame_end).block_index;}
$('#video').ontimeupdate=()=>{$('#position').textContent='时间 '+$('#video').currentTime.toFixed(3)+'s / frame '+frame()+' / block '+block()};
$('#onset').onclick=()=>{$('[data-key="failure_onset_block"]').value=block();$('[data-key="failure_onset_frame"]').value=frame()};
$('#watched').onclick=()=>{$('[data-key="reviewed_until_sec"]').value=$('#video').currentTime.toFixed(4)};
$('#save').onclick=()=>{persist();const quote=x=>'"'+String(x??'').replaceAll('"','""')+'"';const lines=[fields.map(quote).join(','),...records.map(r=>fields.map(k=>quote(r.review[k])).join(','))];const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\r\\n')+'\\r\\n'],{type:'text/csv;charset=utf-8'}));a.download='human_review.csv';a.click();URL.revokeObjectURL(a.href)};show();
</script>'''
    page = page.replace('OPTIONS', options).replace('PAYLOAD', payload).replace('FIELDS', fields)
    page = page.replace('BLOCKMAP', json.dumps(json.loads((run / 'block_map.json').read_text())['blocks']))
    (run / 'review.html').write_text(page)
