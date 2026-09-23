import tempfile
import unittest
from protocol import jobs, prompts, color_texts, COLORS, DESIGN
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize import accuracy, paired_proxy

class ProtocolTests(unittest.TestCase):
    def test_complete_pairing(self):
        a,b=jobs('wan'),jobs('longlive')
        self.assertEqual(len(a),80)
        self.assertEqual({j['pair_id'] for j in a},{j['pair_id'] for j in b})
        self.assertEqual(len({j['video_id'] for j in a+b}),160)
        for x,y in zip(a,b):self.assertEqual(x['prompt'],y['prompt'])
    def test_attribute_only_replacement(self):
        dog=next(p for p in prompts() if p['prompt_id']=='P011')
        target,texts=color_texts(dog)
        self.assertEqual(target,'green')
        self.assertTrue(all('white dog' in t for t in texts))
        self.assertIn('a photo of a white dog wearing a yellow harness',texts)
        self.assertEqual(texts[COLORS.index(target)],dog['attribute_text'])
        for p in prompts():
            c,t=color_texts(p)
            self.assertEqual(t[COLORS.index(c)],p['attribute_text'])
            self.assertEqual(len(set(t)),len(COLORS))
    def test_unknown_not_failure(self):
        a=accuracy(['CORRECT','INCORRECT','UNJUDGEABLE',''])
        self.assertEqual(a['accuracy_among_judgeable'],.5)
        self.assertEqual(a['accuracy_lower_bound'],.25)
        self.assertEqual(a['accuracy_upper_bound'],.75)
        self.assertFalse(a['finalized'])
        self.assertIsNone(accuracy(['',''])['accuracy_among_judgeable'])
        with self.assertRaises(ValueError):accuracy(['probably'])
    def test_paired_denominator(self):
        rows=[dict(pair_id='P001_S00',model='wan',initial_color_proxy_correct='1'),
              dict(pair_id='P001_S00',model='longlive',initial_color_proxy_correct='0'),
              dict(pair_id='P001_S01',model='longlive',initial_color_proxy_correct='1')]
        a=paired_proxy(rows)
        self.assertEqual(a['n'],1)
        self.assertEqual(a['wan_only_correct'],1)
        self.assertEqual(a['longlive_minus_wan'],-1)
    def test_temporal_definition(self):
        self.assertEqual(DESIGN['native_frames'],4*DESIGN['latent_frames']-3)
        self.assertEqual(DESIGN['saved_frames']/DESIGN['fps'],5)
        self.assertEqual(DESIGN['initial_frames'],list(range(16)))
        self.assertTrue(set(DESIGN['proxy_frames']).issubset(DESIGN['initial_frames']))

if __name__=='__main__':unittest.main()
