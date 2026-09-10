"""Four fresh matched branch backwards and base loss, zero optimizer steps."""
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json
from restream.reality_dataset import collate_reality
from restream.reality_encoder import RealityEncoder
from restream.reality_r1 import R1CandidateBank, encode_prefix, select_r1_memory
from restream.reality_paired import load_global_constant
from restream.reality_runtime import read_reality_config, make_memory, make_dataset, prepare_reality, PreserveHistory
from restream.objective import future_loss
from restream.runtime import load_pipeline
from restream.reality_selection import manifest_digest


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def main():
    config = read_reality_config(ROOT/'configs/reality_memory_r1_top1.yaml')
    device = torch.device('cuda')
    ds = make_dataset(config, 'train')
    candidates, refs, donor = R1CandidateBank(ds)[0]
    candidates = candidates.to(device)
    batch = collate_reality([ds[0]])
    cfg = config['reality_memory']
    encoder = RealityEncoder(ROOT/cfg['encoder']['path'], cfg['projector']['memory_dim'],
                             cfg['projector']['num_memory_tokens'], cfg['encoder']['image_size']).to(device).eval().requires_grad_(False)
    assert encoder.identity == ds.cache.identity
    prefix, frame_ids = encode_prefix(encoder, batch['pixels'].to(device), cfg['objective']['prefix_latents'])
    del encoder
    mean = load_global_constant(config, ds.cache, device, 'async')
    pipeline = load_pipeline(config, device)
    gt, cond, anchor = prepare_reality(pipeline, batch, device, config)
    def video(condition):
        return future_loss(pipeline, PreserveHistory(), gt, gt[:, :anchor+1], gt[:, anchor:anchor+1],
                           condition, anchor, torch.Generator(device=device).manual_seed(123), 0)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        base = video(cond).item()
    results, gradients, inputs = {}, {}, {}
    for branch in ('global_async','correct_top1','correct_all2','routed','routed_repeat'):
        torch.manual_seed(config['seed'])
        memory = make_memory(config).to(device)
        initial = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in memory.parameters())).hexdigest()
        features, mask, routing = select_r1_memory('routed' if branch == 'routed_repeat' else branch, prefix, candidates, global_async=mean)
        inputs[branch] = features.cpu()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            fused, _ = memory(cond['prompt_embeds'], features, mask)
            assert torch.equal(fused, cond['prompt_embeds'])
            loss = video({**cond, 'prompt_embeds': fused})
        assert loss.item() == base
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
        assert all(p.grad is None and not p.requires_grad for p in pipeline.parameters())
        gradient = memory.output.weight.grad.detach().float().cpu()
        assert gradient.norm() > 0
        gradients[branch] = gradient
        results[branch] = {'loss': loss.item(), 'base_loss_exact_equal': True, 'initial_adapter_sha256': initial,
                           'memory_shape': list(features.shape), 'memory_sha256': tensor_digest(features),
                           'output_weight_gradient_sha256': tensor_digest(gradient), 'output_weight_gradient_norm': gradient.norm().item(),
                           'parameter_gradient_norms': {n:p.grad.float().norm().item() for n,p in memory.named_parameters()},
                           'routing': {k:v.cpu().tolist() for k,v in routing.items()}}
        print(branch, 'loss',loss.item(),'gradient',gradient.norm().item(),flush=True)
        del loss, fused, memory
    assert len({r['initial_adapter_sha256'] for r in results.values()}) == 1
    assert not torch.equal(gradients['global_async'], gradients['correct_top1'])
    repeat_difference = (gradients['routed_repeat']-gradients['routed']).norm().item()
    comparisons = {}
    for branch in ('global_async','correct_top1','correct_all2'):
        same = inputs[branch].shape == inputs['routed'].shape and torch.equal(inputs[branch], inputs['routed'])
        difference = (gradients[branch]-gradients['routed']).norm().item()
        if same:
            # GPU FlexAttention/BF16 backward uses nondeterministic reductions.
            # Compare against an identical-input repeat, not bitwise equality.
            assert difference <= 3 * repeat_difference + 1e-6
            assert difference / gradients['routed'].norm().item() < .05
        comparisons[branch] = {'same_memory_as_routed': same, 'gradient_difference_norm': (gradients[branch]-gradients['routed']).norm().item()}
    assert (gradients['global_async']-gradients['correct_top1']).norm().item() > 5*repeat_difference
    report = {'status':'passed','optimizer_steps':0,'backward_steps':5,'identical_input_repeat_gradient_difference_norm':repeat_difference,'base_loss':base,
              'sample_id':ds.rows[0]['sample_id'],'noise_seed':123,'adapter_seed':config['seed'],
              'prefix_frames':frame_ids,'branches':results,'comparisons_to_routed':comparisons,
              'train_manifest_sha256':manifest_digest(ROOT/config['data']['train_manifest']),
              'test_eligibility_sha256':manifest_digest(ROOT/'data/r1_test_eligibility.json'),
              'backbone_gradients_none':True,'peak_vram_bytes':torch.cuda.max_memory_allocated(),
              'interpretation':'Identical memory matches within measured BF16/FlexAttention repeat noise; cross-branch difference <=3x repeat norm and <5% gradient norm. Global/correct content gradients differ.'}
    write_json(ROOT/'validation/reality_memory/r1_matched/zero_update_smoke.json',report)


if __name__ == '__main__':
    main()
