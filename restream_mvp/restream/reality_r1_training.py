"""Deterministic matched R1 training/evaluation, with sealed Test excluded."""
import copy
import hashlib
import json
import random
from pathlib import Path
import torch
from .objective import future_loss
from .reality_data import write_json
from .reality_dataset import collate_reality
from .reality_r1 import R1CandidateBank, encode_prefix, select_r1_memory
from .reality_runtime import PreserveHistory, prepare_reality
from .reality_selection import manifest_digest, selection_config_hash, select_targets
from .reality_stats import bootstrap_ci
from .runtime import ROOT

TRAIN_BRANCHES = ('global_async', 'correct_top1', 'routed')
EVAL_BRANCHES = ('none', 'global_async', 'correct_top1', 'correct_all2', 'routed', 'wrong_top1')


def state_digest(model):
    return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def validate_training_config(config):
    cfg = config['reality_memory']
    train = config['train']
    if cfg['stage'] != 'r1a' or cfg['router']['reference_count'] != 2:
        raise ValueError('Matched R1 requires correct 2 + wrong 2 candidates')
    if not .2 <= train['no_memory_probability'] <= .3:
        raise ValueError('R1 No-Memory dropout must be 20–30%')
    if cfg['references']['per_reference_dropout'] != 0 or cfg['regularization']['wrong_gate_weight'] != 0 or 'paired' in cfg['objective']:
        raise ValueError('R1 uses only whole-memory dropout, video loss and delta regularization')
    if not 0 <= cfg['regularization']['delta_weight'] < float('inf'):
        raise ValueError('Invalid delta weight')
    if type(train['max_updates']) is not int or not 2 <= train['max_updates'] <= 30:
        raise ValueError('This bounded R1 trainer supports at most 30 updates')
    if type(config['eval']['cases']) is not int or config['eval']['cases'] < 1:
        raise ValueError('Fix a positive Dev subset size before training')
    seeds = config['eval']['noise_seeds']
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Fix distinct nonnegative Dev noise seeds')


def sealed_test_provenance(config):
    report_path = ROOT/config['data']['test_eligibility']
    report = json.loads(report_path.read_text())
    policy_path = ROOT/'data/r1_test_eligibility_policy.json'
    if report['status'] != 'sealed' or report['models_run'] or report['source_replacements']:
        raise ValueError('Test eligibility must be sealed before training, without model decisions')
    if manifest_digest(ROOT/report['manifest']) != report['manifest_sha256'] or manifest_digest(policy_path) != report['policy_sha256']:
        raise ValueError('Sealed Test artifact changed')
    policy = json.loads(policy_path.read_text())
    if report['selection_identity'] != policy['selection_identity'] or policy['selection_config_hash'] != selection_config_hash(config):
        raise ValueError('Training data rules differ from sealed Test eligibility')
    if policy['input_sha256']['data/r1_split_lock.json'] != manifest_digest(ROOT/config['data']['r1_split_lock']):
        raise ValueError('Test seal belongs to another source split')
    return {'report_sha256': manifest_digest(report_path), 'policy_sha256':report['policy_sha256'],
            'test_manifest_sha256':report['manifest_sha256'],'eligible_targets':report['eligible_targets']}


def schedule_entry(rows, batch_step, seed, dropout):
    """Stateless RNG namespaces keep branches/resume/dropout perfectly matched."""
    epoch, offset = divmod(batch_step, len(rows))
    order = list(range(len(rows)))
    random.Random(f'order:{seed}:{epoch}').shuffle(order)
    return {'index':order[offset], 'no_memory':random.Random(f'drop:{seed}:{batch_step}').random() < dropout,
            'noise_seed':random.Random(f'noise:{seed}:{batch_step}').randrange(2**31)}


class PreparedInputs:
    """Cache only frozen inputs in CPU RAM; never optimizer-dependent activations."""
    def __init__(self, dataset, encoder, pipeline, config, device):
        self.dataset, self.encoder, self.pipeline = dataset, encoder, pipeline
        self.config, self.device = config, device
        self.bank = R1CandidateBank(dataset, config['reality_memory']['router']['reference_count'])
        self.cache = {}

    def get(self, index):
        if index not in self.cache:
            batch = collate_reality([self.dataset[index]])
            cfg = self.config['reality_memory']
            prefix, frames = encode_prefix(self.encoder, batch['pixels'].to(self.device),
                                            cfg['objective']['prefix_latents'], cfg['router']['prefix_frames'])
            gt, cond, anchor = prepare_reality(self.pipeline, batch, self.device, self.config)
            candidates, refs, donor = self.bank[index]
            self.cache[index] = {'gt':gt.cpu(), 'cond':{k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in cond.items()},
                                 'anchor':anchor,'prefix':prefix.cpu(),'candidates':candidates,
                                 'sample_id':self.dataset.rows[index]['sample_id'],
                                 'donor_sample_id':self.dataset.rows[donor]['sample_id'],'prefix_frames':frames}
        record = self.cache[index]
        return {**record,'gt':record['gt'].to(self.device),'prefix':record['prefix'].to(self.device),
                'candidates':record['candidates'].to(self.device),
                'cond':{k:v.to(self.device) if isinstance(v,torch.Tensor) else v for k,v in record['cond'].items()}}


def r1_loss(pipeline, memory, item, branch, global_async, noise_seed, delta_weight, no_memory=False):
    features, mask, routing = select_r1_memory(branch, item['prefix'], item['candidates'], global_async=global_async)
    if no_memory:
        mask = torch.zeros_like(mask)
    fused, stats = memory(item['cond']['prompt_embeds'], features, mask)
    gt, anchor = item['gt'], item['anchor']
    video = future_loss(pipeline, PreserveHistory(), gt, gt[:,:anchor+1], gt[:,anchor:anchor+1],
                        {**item['cond'],'prompt_embeds':fused}, anchor,
                        torch.Generator(device=gt.device).manual_seed(noise_seed), 0)
    return video + delta_weight*stats['delta_square'], {**stats,'video_loss':video.detach(),'routing':routing}


def optimizer_update(memory, opt, scheduler, no_memory, clip):
    """All-drop batches must never run AdamW (including decay/momentum)."""
    params = list(memory.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise RuntimeError('Missing/nonfinite adapter gradients')
    norms = {name:p.grad.float().norm().item() for name,p in memory.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(params, clip)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite global gradient norm')
    if no_memory:
        if norm.item() != 0:
            raise RuntimeError('No-Memory batch has nonzero adapter gradients')
        return False, norms, norm.item()
    if norm.item() == 0:
        raise RuntimeError('Active memory produced no gradient')
    opt.step()
    scheduler.step()
    return True, norms, norm.item()


def run_signature(config, cache, branch):
    semantic = copy.deepcopy(config)
    semantic['train'].pop('max_updates', None)
    return {'config':semantic,'branch':branch,'encoder':cache.identity,
            'manifests':{s:manifest_digest(ROOT/config['data'][f'{s}_manifest']) for s in ('train','val')},
            'split_lock':manifest_digest(ROOT/config['data']['r1_split_lock']),
            'global_mean':manifest_digest(ROOT/config['reality_memory']['references']['global_constant_features']),
            'test_seal':sealed_test_provenance(config),
            'implementation':{p:manifest_digest(ROOT/p) for p in (
                'train_reality_r1.py','restream/reality_r1_training.py','restream/reality_r1.py',
                'restream/reality_router.py','restream/reality_memory.py','restream/objective.py',
                'restream/reality_runtime.py','restream/runtime.py')}}


@torch.no_grad()
def evaluate(pipeline, memory, prepared, global_async, config, output):
    mode = memory.training
    memory.eval()
    cfg = config['eval']
    indices = select_targets(prepared.dataset.rows, cfg['cases'], cfg['target_seed'])
    entries = []
    try:
        for index in indices:
            item = prepared.get(index)
            for seed in cfg['noise_seeds']:
                variants = {}
                for branch in EVAL_BRANCHES:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss, stats = r1_loss(pipeline,memory,item,branch,global_async,seed,0)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite Dev video loss')
                    variants[branch] = {'video_loss':loss.item(),'gate':stats['gate'].mean().item(),
                                        'selected_indices':stats['routing'].get('selected_indices',torch.empty(0,dtype=torch.long)).cpu().tolist()}
                entries.append({'sample_id':item['sample_id'],'noise_seed':seed,'variants':variants})
            print(f"Dev {len(entries)}/{len(indices)*len(cfg['noise_seeds'])}: {item['sample_id']}",flush=True)
    finally:
        memory.train(mode)
    per_target = {}
    for row in entries:
        v = {k:x['video_loss'] for k,x in row['variants'].items()}
        per_target.setdefault(row['sample_id'],[]).append({
            'G_content_matched':v['global_async']-v['correct_top1'],
            'S_route':v['wrong_top1']-v['routed'],
            'routed_minus_oracle':v['routed']-v['correct_top1'],
            'routed_gain_over_global':v['global_async']-v['routed']})
    metrics = {k:bootstrap_ci([sum(r[k] for r in records)/len(records) for records in per_target.values()],2000,cfg['target_seed'])
               for k in next(iter(per_target.values()))[0]}
    report = {'split':'dev','indices':indices,'target_seed':cfg['target_seed'],'noise_seeds':cfg['noise_seeds'],
              'unique_targets':len(indices),'cases':entries,'metrics':metrics,
              'mean_video_loss':{branch:sum(r['variants'][branch]['video_loss'] for r in entries)/len(entries) for branch in EVAL_BRANCHES},
              'scope':'Fixed preliminary Dev subset, not untouched Test; bootstrap over targets, not noise seeds.'}
    write_json(output,report)
    return report
