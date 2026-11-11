"""Shared mechanics for the separately scoped Qwen3.5-2B pilot."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HOME']=str(ROOT/'data/stair/hf-qwen-cache')
from collections import Counter
from datetime import datetime,timezone
import gc,hashlib,json,random,threading,time,unicodedata
import numpy as np
import psutil
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer,Qwen3_5ForCausalLM,Qwen3_5TextConfig,LogitsProcessor,LogitsProcessorList
from peft import LoraConfig,TaskType,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
from safetensors.torch import load_file

MODEL_FOLDER=ROOT/'data/stair/qwen35_2b_model'
MODEL='Qwen/Qwen3.5-2B'
REVISION='15852e8c16360a2fea060d615a32b45270f8a8fc'
BASE_RUN=ROOT/'runs/stair/improvement-pilots/cheap-recovery-20261107T104810314921Z'
TARGET_MODULES=['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','in_proj_b','in_proj_a','out_proj','gate_proj','up_proj','down_proj']


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    def encode(item):
        if isinstance(item,set):return sorted(item)
        raise TypeError('Unsupported evidence value: '+type(item).__name__)
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=encode)+'\n');temp.replace(path)
def records(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def append(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def canonical(s):return ' '.join(unicodedata.normalize('NFC',s).split())
def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.backends.mps.is_available():torch.mps.manual_seed(42)
def sync(device='mps'):
    if str(device)=='mps':torch.mps.synchronize()
def release():
    gc.collect()
    if torch.backends.mps.is_available():torch.mps.empty_cache()
def block_network():
    import socket
    def forbidden(*a,**k):raise RuntimeError('Network forbidden inside cached-input pilot')
    socket.socket.connect=forbidden;socket.create_connection=forbidden
def setup():
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required; CPU fallback forbidden')
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);seed()


class Guard:
    def __init__(self,folder,seconds,prior_seconds=0):
        self.folder=Path(folder);self.seconds=seconds;self.prior_seconds=prior_seconds
        self.started=time.monotonic();self.wall=time.time();self.phase='setup';self.samples=[]
        self.swap_start=psutil.swap_memory().used;self.stop=threading.Event()
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()
    def elapsed(self):return self.prior_seconds+time.monotonic()-self.started
    def sample(self):
        r={'elapsed_seconds':self.elapsed(),'wall_elapsed_seconds':time.time()-self.wall,'stage':self.phase,
            'rss_bytes':psutil.Process().memory_info().rss,'available_bytes':psutil.virtual_memory().available,
            'swap_bytes':psutil.swap_memory().used,'mps_driver_bytes':torch.mps.driver_allocated_memory(),
            'mps_tensor_bytes':torch.mps.current_allocated_memory(),'mps_recommended_bytes':torch.mps.recommended_max_memory()}
        self.samples.append(r);reason=None
        if r['rss_bytes']>48*2**30 or r['available_bytes']<32*2**30:reason='RSS/available memory limit'
        if r['mps_driver_bytes']>min(48*2**30,.8*r['mps_recommended_bytes']):reason='MPS driver limit'
        if r['swap_bytes']-self.swap_start>2**30:reason='Swap-growth limit'
        if self.elapsed()>self.seconds:reason='Measured compute deadline'
        if reason:
            write(self.folder/'resource_violation.json',{'reason':reason,'sample':r})
            write(self.folder/'memory_samples.json',self.samples);os._exit(70)
    def watch(self):
        while not self.stop.wait(.5):self.sample()
    def event(self,kind,**fields):
        row={'kind':kind,'timestamp':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':self.elapsed(),**fields}
        append(self.folder/'events.jsonl',row);print(json.dumps(row),flush=True)
    def close(self):
        self.stop.set();self.thread.join(timeout=2);write(self.folder/'memory_samples.json',self.samples)


def prompt(fmt,query,hierarchy):
    context='' if fmt=='D' else hierarchy['flat_text'] if fmt=='E' else hierarchy['hierarchical_text']
    return 'Return exactly one document title from Principles of Financial Accounting.\n'+context+'\nQuestion: '+query+'\nTitle:'


def tokenize_data(data):
    tok=AutoTokenizer.from_pretrained(MODEL_FOLDER,local_files_only=True);tok.padding_side='left'
    if tok.eos_token!='<|im_end|>' or tok.pad_token_id is None:raise ValueError('Unexpected chat EOS/padding')
    paths=[tok.encode(canonical(r['title']),add_special_tokens=False)+[tok.eos_token_id] for r in data['corpus']]
    if len({tuple(p) for p in paths})!=86 or any(tok.decode(p[:-1],clean_up_tokenization_spaces=False)!=canonical(r['title']) for p,r in zip(paths,data['corpus'])):
        raise ValueError('Title roundtrip/injectivity failed')
    encoded={};summary={}
    for fmt in 'DEF':
        encoded[fmt]={}
        for split in ['train935','tune','screen']:
            rows=[]
            for raw in data[split]:
                text=tok.apply_chat_template([{'role':'user','content':prompt(fmt,raw['query'],data['hierarchy'])}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=tok.encode(text,add_special_tokens=False)
                target=tok.encode(raw['gold_title'],add_special_tokens=False)+[tok.eos_token_id]
                if len(ids)+len(target)>4096 or len(target)>96:raise ValueError('Complete-input/target cap exceeded')
                rows.append({'query_id':raw['query_id'],'prompt_ids':ids,'target':target,'ids':ids+target,
                    'prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
            encoded[fmt][split]=rows
        lengths=[len(r['prompt_ids']) for rows in encoded[fmt].values() for r in rows]
        summary[fmt]={'min':min(lengths),'median':float(np.median(lengths)),'max':max(lengths)}
    return tok,paths,encoded,{'complete_prompt_tokens':summary,'target_roundtrip':'passed','target_injectivity':'passed',
        'targets':86,'max_target_tokens':max(map(len,paths)),'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id,'thinking':False}


def load_model():
    seed();config=Qwen3_5TextConfig.from_pretrained(MODEL_FOLDER,local_files_only=True)
    model,info=Qwen3_5ForCausalLM.from_pretrained(MODEL_FOLDER,config=config,local_files_only=True,dtype=torch.bfloat16,
        device_map={'':'mps'},attn_implementation='eager',output_loading_info=True)
    sync()
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):raise ValueError('Language checkpoint incomplete')
    n=sum(p.numel() for p in model.parameters())
    if n!=1881825088 or any(p.device.type!='mps' or p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('2B base placement/size/dtype failed')
    modules=[name for name,layer in model.named_modules() if isinstance(layer,torch.nn.Linear) and name.split('.')[-1] in TARGET_MODULES]
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=0.,bias='none',task_type=TaskType.CAUSAL_LM,target_modules=modules))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    adapters=[p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in adapters)!=16819200 or any(p.dtype!=torch.float32 or p.device.type!='mps' for p in adapters):raise ValueError('Adapter placement/count failed')
    initial={k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(model).items()}
    sample={n:p.flatten()[:16].detach().cpu().clone() for n,p in model.named_parameters() if not p.requires_grad and ('embed_tokens' in n or 'layers.0.' in n or 'layers.23.' in n)}
    return model,initial,sample,{'loading_info':info,'base_parameters':n,'adapter_parameters':sum(p.numel() for p in adapters),'modules':modules}


def optimizer(model):return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
def reset(model,initial):
    set_peft_model_state_dict(model,initial);model.zero_grad(set_to_none=True);seed()
    return optimizer(model)


def collate(rows,pad,device):
    width=max(len(r['ids']) for r in rows);target_width=max(len(r['target']) for r in rows)
    ids=torch.full((len(rows),width),pad,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids);labels=torch.full_like(ids,-100)
    for i,row in enumerate(rows):
        ids[i,-len(row['ids']):]=torch.tensor(row['ids'],device=device)
        mask[i,-len(row['ids']):]=1
        labels[i,-len(row['target']):]=torch.tensor(row['target'],device=device)
    positions=(mask.cumsum(-1)-1).masked_fill(mask==0,0)
    return {'input_ids':ids,'attention_mask':mask,'position_ids':positions},labels[:,-target_width:],target_width


def example_losses(model,rows,pad,device='mps'):
    batch,labels,width=collate(rows,pad,device)
    output=model(**batch,use_cache=False,logits_to_keep=width+1)
    loss=F.cross_entropy(output.logits[:,:-1,:].float().transpose(1,2),labels,ignore_index=-100,reduction='none')
    return loss.sum(-1)/(labels!=-100).sum(-1)


def update(model,opt,rows,pad,microbatch,device='mps',return_gradient=False):
    model.train();model.config.use_cache=False;opt.zero_grad(set_to_none=True)
    sync(device);tick=time.monotonic();losses=[]
    for start in range(0,len(rows),microbatch):
        values=example_losses(model,rows[start:start+microbatch],pad,device)
        if not bool(torch.isfinite(values).all()):raise ValueError('Nonfinite training loss')
        (values.sum()/len(rows)).backward();losses.extend(values.detach().cpu().tolist());del values
    trainable=[p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen base has gradients')
    if not any(p.grad is not None for p in trainable) or any(p.grad is not None and p.grad.device.type!=str(device) for p in trainable):raise ValueError('Gradient placement failed')
    norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True,foreach=False)
    if not bool(torch.isfinite(norm)) or float(norm)==0:raise ValueError('Invalid/zero gradient')
    grad=torch.cat([p.grad.detach().float().cpu().flatten() for p in trainable if p.grad is not None]) if return_gradient else None
    opt.step();sync(device)
    return {'loss':float(np.mean(losses)),'gradient_norm':float(norm),'seconds':time.monotonic()-tick},grad


class CausalTrie(LogitsProcessor):
    def __init__(self,paths,prefix_width,eos,pad):
        self.root={};self.width=prefix_width;self.eos=eos;self.pad=pad
        for path in paths:
            node=self.root
            for token in path:node=node.setdefault(int(token),{})
    def allowed(self,generated):
        node=self.root
        for i,token in enumerate(generated):
            if token not in node:raise ValueError('Generated prefix left title trie')
            node=node[token]
            if token==self.eos:
                if any(t not in [self.pad,self.eos] for t in generated[i+1:]):raise ValueError('Tokens after complete title')
                return [self.eos]
        return sorted(node)
    def __call__(self,input_ids,scores):
        if not torch.allclose(torch.logsumexp(scores,dim=-1),torch.zeros(scores.shape[0],device=scores.device),atol=1e-3,rtol=1e-3):raise ValueError('Expected full-vocabulary beam log probabilities')
        masked=torch.full_like(scores,-float('inf'))
        for i,ids in enumerate(input_ids.tolist()):masked[i,self.allowed(ids[self.width:])]=scores[i,self.allowed(ids[self.width:])]
        return masked


def decode(model,tok,paths,rows,corpus,fmt,mode,device='mps'):
    tick=time.monotonic();width=max(len(r['prompt_ids']) for r in rows)
    ids=torch.full((len(rows),width),tok.pad_token_id,dtype=torch.long,device=device);mask=torch.zeros_like(ids)
    for i,row in enumerate(rows):
        ids[i,-len(row['prompt_ids']):]=torch.tensor(row['prompt_ids'],device=device);mask[i,-len(row['prompt_ids']):]=1
    processors=LogitsProcessorList([CausalTrie(paths,width,tok.eos_token_id,tok.pad_token_id)]) if mode=='constrained' else LogitsProcessorList()
    model.eval();model.config.use_cache=True;sync(device);started=time.monotonic()
    with torch.inference_mode():
        out=model.generate(input_ids=ids,attention_mask=mask,num_beams=4,num_return_sequences=3,max_new_tokens=96,
            do_sample=False,length_penalty=1.,early_stopping='never',renormalize_logits=False,forced_eos_token_id=None,
            pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=True,
            return_dict_in_generate=True,output_scores=True,logits_processor=processors)
        transitions=model.compute_transition_scores(out.sequences,out.scores,out.beam_indices,normalize_logits=False)
    sync(device);model_seconds=time.monotonic()-started
    sequences=out.sequences[:,width:].cpu().tolist();transitions=transitions.cpu().tolist();sequence_scores=out.sequences_scores.cpu().tolist()
    mapping={canonical(r['title']):f'finance:{i:03d}' for i,r in enumerate(corpus)};result=[]
    for i,row in enumerate(rows):
        hypotheses=[]
        for rank in range(3):
            index=i*3+rank;tokens=sequences[index];complete=tok.eos_token_id in tokens
            n=tokens.index(tok.eos_token_id)+1 if complete else len(tokens);tokens=tokens[:n];logp=transitions[index][:n]
            score=float(sum(logp)/max(1,n))
            if len(logp)!=n or not np.isclose(score,sequence_scores[index],atol=1e-3,rtol=1e-3):raise ValueError('Causal beam score reconstruction failed')
            text=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False);title=canonical(text)
            sid=mapping.get(title) if complete else None
            validity='valid' if sid else 'incomplete' if not complete else 'empty' if not title else 'nonexistent_title'
            if mode=='constrained' and not sid:raise ValueError('Invalid constrained title')
            hypotheses.append({'rank':rank+1,'section_id':sid,'raw_text':text,'canonical_title':title,'validity':validity,
                'complete_eos':complete,'token_ids':tokens,'token_logprobs':logp,'score':score})
        hypotheses.sort(key=lambda h:(-h['score'],(0,h['section_id']) if h['section_id'] else (1,tuple(h['token_ids']))))
        for rank,h in enumerate(hypotheses,1):h['rank']=rank
        result.append({'query_id':row['query_id'],'method':fmt,'mode':mode,'prompt_sha256':row['prompt_sha256'],
            'input_tokens':len(row['prompt_ids']),'hypotheses':hypotheses})
    timing={'batch_size':len(rows),'model_seconds':model_seconds,'end_to_end_seconds':time.monotonic()-tick}
    for row in result:row['timing']=timing
    return result


def infer(model,tok,paths,rows,corpus,fmt,mode,batch,guard=None):
    result=[]
    for start in range(0,len(rows),batch):
        result.extend(decode(model,tok,paths,rows[start:start+batch],corpus,fmt,mode))
        if guard and (start+batch)%32==0:guard.event('inference_progress',format=fmt,mode=mode,queries=min(start+batch,len(rows)))
    return result
def equal(left,right):
    return len(left)==len(right) and all(a['query_id']==b['query_id'] and
        [h['token_ids'] for h in a['hypotheses']]==[h['token_ids'] for h in b['hypotheses']] and
        [h['validity'] for h in a['hypotheses']]==[h['validity'] for h in b['hypotheses']] and
        np.allclose([h['score'] for h in a['hypotheses']],[h['score'] for h in b['hypotheses']],atol=1e-3,rtol=1e-3)
        for a,b in zip(left,right))
def quality(predictions,queries,corpus):
    gold={r['query_id']:r['gold_title'] for r in queries};lookup={canonical(r['title']):f'finance:{i:03d}' for i,r in enumerate(corpus)}
    if len(predictions)!=len(gold) or {r['query_id'] for r in predictions}!=set(gold):raise ValueError('Prediction membership changed')
    scores=[]
    for row in predictions:
        ranks=[i for i,h in enumerate(row['hypotheses'][:3],1) if h['section_id']==lookup[gold[row['query_id']]]]
        r=min(ranks) if ranks else None
        scores.append((int(r==1),int(r is not None),1/np.log2(r+1) if r else 0,int(row['hypotheses'][0]['validity']!='valid')))
    means=np.mean(scores,axis=0)
    return {'queries':len(scores),**{k:float(v) for k,v in zip(['r1','r3','ndcg3','invalid1'],means)}}
def groups(rows):
    pending=[];epoch=1
    while True:
        for index in np.random.default_rng(42002+epoch).permutation(len(rows)):
            pending.append(rows[int(index)])
            if len(pending)==8:yield pending;pending=[]
        epoch+=1
