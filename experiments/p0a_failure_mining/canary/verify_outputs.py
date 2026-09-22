import argparse,json,sys
from pathlib import Path
import av,torch
from PIL import Image,ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import digest,write_json
parser=argparse.ArgumentParser(description='Validate every Canary MP4 and replay capsule, with two-second overview sheets.')
parser.add_argument('--run-dir',type=Path,required=True)
run=parser.parse_args().run_dir
rows=[]
for complete in sorted((run/'outputs').glob('*/complete.json')):
 row=json.loads(complete.read_text());out=complete.parent
 for name,expected in row['artifact_sha256'].items():
  assert digest(out/name)==expected,(out,name)
 with av.open(str(out/'video.mp4')) as c:
  st=c.streams.video[0]
  assert float(st.average_rate)==16 and st.width==832 and st.height==480
  times=[]; images=[]
  for i,frame in enumerate(c.decode(video=0)):
   times.append(float(frame.pts*frame.time_base))
   if i%32==0:
    im=frame.to_image();im.thumbnail((208,120));images.append((i,im))
  assert len(times)==960 and all(abs(t-i/16)<1e-6 for i,t in enumerate(times))
 sheet=Image.new('RGB',(1040,150*6),'white');d=ImageDraw.Draw(sheet)
 for j,(i,im) in enumerate(images):
  x,y=j%5*208,j//5*150;sheet.paste(im,(x,y));d.text((x+3,y+123),f'{i/16:.0f}s / frame {i}',fill='black')
 sheet.save(out/'overview_2s.jpg')
 capsule=torch.load(out/'replay.pt',map_location='cpu',weights_only=True)
 assert capsule['plan_signature']==row['plan_signature']
 assert capsule['noise'].shape==capsule['latents'].shape==(1,243,16,60,104)
 assert capsule['latents'].dtype==torch.bfloat16 and torch.isfinite(capsule['latents']).all()
 assert len(capsule['boundary_rng'])==81
 assert all(len(s['cuda'])==len(capsule['initial_rng']['cuda']) and s['cpu'].dtype==torch.uint8 for s in capsule['boundary_rng'])
 result=dict(video_id=row['video_id'],decoded_frames=960,duration_sec=60,fps=16,width=832,height=480,
             num_blocks=81,replay_latents_finite=True,replay_rng_boundaries=81,artifacts_verified=True,
             total_sec=row['total_sec'],generation_and_save_sec=row['generation_and_save_sec'],
             peak_cuda_allocated_gib=row['peak_cuda_allocated_bytes']/2**30,
             peak_cuda_reserved_gib=row['peak_cuda_reserved_bytes']/2**30,
             video_bytes=(out/'video.mp4').stat().st_size,replay_bytes=(out/'replay.pt').stat().st_size)
 rows.append(result);print(result,flush=True)
write_json(run/'manifests/generation_validation.json',{'videos':rows,'count':len(rows),'expected':4,'all_four_complete':len(rows)==4})
