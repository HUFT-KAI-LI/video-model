import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from restream.reality_memory import RealityMemory
from restream.reality_r1 import select_r1_memory
from restream.reality_r1_training import optimizer_update, schedule_entry, validate_training_config, sealed_test_provenance
from restream.reality_runtime import read_reality_config
from scripts.seal_r1_test import immutable_write


class R1TrainingTests(unittest.TestCase):
    def test_k_matched_oracle_keeps_same_choice_when_distractors_lose(self):
        candidates=torch.tensor([[[[1.,0.,0.,0.]],[[0.,1.,0.,0.]], [[-1.,0.,0.,0.]],[[0.,-1.,0.,0.]]]])
        prefix=candidates[:,1]
        routed=select_r1_memory('routed',prefix,candidates)
        oracle=select_r1_memory('correct_top1',prefix,candidates)
        wrong=select_r1_memory('wrong_top1',prefix,candidates)
        global_=select_r1_memory('global_async',prefix,candidates,global_async=torch.ones(1,4))
        for result in (routed,oracle,wrong,global_):
            self.assertEqual(result[0].shape,(1,1,1,4))
        torch.testing.assert_close(routed[0],oracle[0],rtol=0,atol=0)
        self.assertEqual(oracle[2]['selected_indices'].item(),1)
        self.assertGreaterEqual(wrong[2]['selected_indices'].item(),2)
        none=select_r1_memory('none',prefix,candidates)
        self.assertFalse(none[1].any())

    def test_distractor_can_change_only_routed_choice(self):
        prefix=torch.tensor([[[1.,0.]]])
        bank=torch.tensor([[[[.7,.7]],[[.2,.8]],[[1.,0.]],[[0.,1.]]]])
        self.assertEqual(select_r1_memory('routed',prefix,bank)[2]['selected_indices'].item(),2)
        self.assertEqual(select_r1_memory('correct_top1',prefix,bank)[2]['selected_indices'].item(),0)

    def test_no_memory_does_not_apply_adamw_decay_or_advance_scheduler(self):
        model=RealityMemory(4,4,8,1)
        torch.nn.init.normal_(model.output.weight)
        opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:1/(step+1))
        context=torch.randn(1,3,8);features=torch.randn(1,1,2,4)
        before={k:v.clone() for k,v in model.state_dict().items()}
        fused,stats=model(context,features,torch.zeros(1,1,dtype=torch.bool))
        (fused.square().mean()+.001*stats['delta_square']).backward()
        epoch=scheduler.last_epoch
        updated,_,norm=optimizer_update(model,opt,scheduler,True,1.)
        self.assertFalse(updated);self.assertEqual(norm,0);self.assertEqual(scheduler.last_epoch,epoch)
        self.assertFalse(opt.state)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_stateless_schedule_and_checkpoint_resume_match_uninterrupted(self):
        rows=list(range(10));seed=42
        schedule=[schedule_entry(rows,i,seed,.25) for i in range(20)]
        self.assertTrue(any(r['no_memory'] for r in schedule))
        self.assertEqual(schedule[7:],[schedule_entry(rows,i,seed,.25) for i in range(7,20)])
        def fresh():
            torch.manual_seed(7)
            model=torch.nn.Linear(2,1)
            opt=torch.optim.AdamW(model.parameters(),lr=.01)
            sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1.,(step+1)/10))
            return model,opt,sch
        def step(triple,i):
            m,o,s=triple;o.zero_grad(set_to_none=True)
            e=schedule[i]
            pred=m(torch.tensor([[float(e['index']),1.]]))
            (pred.square().mean() * (0 if e['no_memory'] else 1)).backward()
            optimizer_update(m,o,s,e['no_memory'],1.)
        a=fresh()
        for i in range(20):step(a,i)
        b=fresh()
        for i in range(7):step(b,i)
        data=io.BytesIO();torch.save([x.state_dict() for x in b],data);data.seek(0)
        states=torch.load(data,weights_only=False);c=fresh()
        for obj,state in zip(c,states):obj.load_state_dict(state)
        for i in range(7,20):step(c,i)
        for x,y in zip(a[0].parameters(),c[0].parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual(a[2].state_dict(),c[2].state_dict())

    def test_seal_and_training_configuration(self):
        config=read_reality_config(ROOT/'configs/reality_memory_r1_matched.yaml')
        validate_training_config(config)
        self.assertEqual(sealed_test_provenance(config)['eligible_targets'],0)
        for rate in (0.,.5):
            bad=copy.deepcopy(config);bad['train']['no_memory_probability']=rate
            with self.assertRaises(ValueError):validate_training_config(bad)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'sealed.jsonl'
            immutable_write(path,'');immutable_write(path,'')
            with self.assertRaises(FileExistsError):immutable_write(path,'changed')


if __name__=='__main__': unittest.main()
