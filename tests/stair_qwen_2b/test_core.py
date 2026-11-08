"""Small CPU fixtures for causal alignment, batching and trie completion."""
import copy
import importlib.util
from pathlib import Path
import unittest
import tempfile

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('qwen_core',ROOT/'notebook_src/stair_qwen_2b.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def tiny():
    m.torch.manual_seed(11)
    c=m.Qwen3_5TextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=2,num_key_value_heads=1,head_dim=8,layer_types=['linear_attention','full_attention'],
        linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=4,linear_value_head_dim=4,
        linear_conv_kernel_dim=4,tie_word_embeddings=True,eos_token_id=1,pad_token_id=0,
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':.5,
            'mrope_section':[1,1,0],'mrope_interleaved':True})
    return m.Qwen3_5ForCausalLM(c).float()


class QwenCoreTests(unittest.TestCase):
    def test_loader_metadata_sets_are_saved_deterministically(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'loading.json'
            m.write(path,{'unexpected_keys':{'vision.b','vision.a'}})
            self.assertEqual(m.read(path),{'unexpected_keys':['vision.a','vision.b']})

    def test_target_only_projection_matches_independent_full_causal_loss(self):
        model=tiny().eval()
        rows=[{'ids':[7,8,2,1],'target':[2,1]}, {'ids':[9,10,11,3,4,1],'target':[3,4,1]}]
        with m.torch.no_grad():
            actual=m.example_losses(model,rows,0,'cpu')
            expected=[]
            for row in rows:
                ids=m.torch.tensor([row['ids']])
                logits=model(input_ids=ids,use_cache=False).logits
                start=len(row['ids'])-len(row['target'])-1
                expected.append(m.F.cross_entropy(logits[0,start:-1].float(),m.torch.tensor(row['target'])))
        self.assertTrue(m.torch.allclose(actual,m.torch.stack(expected),atol=1e-5,rtol=1e-5))

    def test_accumulation_matches_full_group_gradient(self):
        one=tiny();two=copy.deepcopy(one)
        rows=[{'ids':[7,8,2,1],'target':[2,1]}, {'ids':[9,10,11,3,4,1],'target':[3,4,1]}]
        a,_=m.update(one,m.torch.optim.SGD(one.parameters(),lr=.01),rows,0,1,'cpu')
        b,_=m.update(two,m.torch.optim.SGD(two.parameters(),lr=.01),rows,0,2,'cpu')
        self.assertAlmostEqual(a['loss'],b['loss'],places=5)
        self.assertTrue(all(m.torch.allclose(x,y,atol=1e-6,rtol=1e-5) for x,y in zip(one.parameters(),two.parameters())))

    def test_trie_prefix_eos_and_post_eos_padding(self):
        trie=m.CausalTrie([[2,1],[2,3,1],[4,1]],12,1,0)
        self.assertEqual(trie.allowed([2]),[1,3])
        self.assertEqual(trie.allowed([2,1,0]),[1])
        with self.assertRaises(ValueError):trie.allowed([2,1,7])
        with self.assertRaises(ValueError):trie.allowed([9])


if __name__=='__main__':unittest.main()
