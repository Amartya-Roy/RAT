#!/usr/bin/env python3
"""RAT_main.py — Recursive Arithmetic Transformer.

Requires Python >=3.10, PyTorch >=2.1.

Quick start:
  python RAT_main.py train --out runs/rat_seed42.pt --seed 42 --steps 10000
  python RAT_main.py evaluate --checkpoint runs/rat_seed42.pt --n 3 4 5 6 7 8 9 10 --examples 1000
  python RAT_main.py predict --checkpoint runs/rat_seed42.pt --a 1234 --b 5678
  python RAT_main.py predict --a 1234 --b 5678 --oracle-all
  python RAT_main.py self-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

TOKENS = ["<pad>", "<bos>", "<eos>", "<split>", "<call>", "<combine>", "|", "=", "+", "*", *list("0123456789")]
TO_ID = {tok: i for i, tok in enumerate(TOKENS)}
DIGIT_BASE = TO_ID["0"]
DIGIT_IDS = set(range(DIGIT_BASE, DIGIT_BASE + 10))
PAD = TO_ID["<pad>"]
BOS = TO_ID["<bos>"]
EOS = TO_ID["<eos>"]
SPLIT = TO_ID["<split>"]
CALL = TO_ID["<call>"]
COMBINE = TO_ID["<combine>"]
BAR = TO_ID["|"]
EQUAL = TO_ID["="]
PLUS = TO_ID["+"]
STAR = TO_ID["*"]


def digits_len(n: int) -> int:
    return len(str(n))


def num_digits(value: int, order: str = "lsd", width: Optional[int] = None) -> list[int]:
    if value < 0:
        raise ValueError("Only nonnegative integers are supported")
    digits = list(str(value).zfill(width or 1))
    if order == "lsd":
        digits.reverse()
    elif order != "msd":
        raise ValueError("order must be 'lsd' or 'msd'")
    return [DIGIT_BASE + int(d) for d in digits]


def parse_digits(tokens: Sequence[int], order: str = "lsd") -> Optional[int]:
    if not tokens or any(t not in DIGIT_IDS for t in tokens):
        return None
    s = ''.join(str(t - DIGIT_BASE) for t in tokens)
    if order == "lsd":
        s = s[::-1]
    return int(s)


def sample_number(rng: random.Random, digits: int, *, exact: bool = True) -> int:
    if digits <= 0:
        raise ValueError("Operand digits must be positive")
    lo = 10 ** (digits - 1) if exact else 0
    return rng.randrange(lo, 10 ** digits)


def split_parts(a: int, b: int) -> tuple[int, list[tuple[int, int]]]:
    length = max(digits_len(a), digits_len(b))
    if length <= 1:
        raise ValueError("Cannot split length <= 1")
    k = math.ceil(length / 2)
    base = 10 ** k
    al, ah = a % base, a // base
    bl, bh = b % base, b // base
    return k, [(al, bl), (al, bh), (ah, bl), (ah, bh)]


def combine_numeric(parts: Sequence[int], k: int) -> int:
    if len(parts) != 4:
        raise ValueError("Requires four child products")
    return parts[0] + (parts[1] + parts[2]) * (10 ** k) + parts[3] * (10 ** (2 * k))


@dataclass
class Config:
    layers: int = 6
    d_model: int = 256
    heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0
    place_max: int = 512
    max_seq: int = 1024
    base_cutoff: int = 2
    split_train_max: int = 8
    split_copy_max: int = 32
    combine_k_max: int = 16
    combine_r_max: int = 32
    batch_size: int = 64
    lr: float = 8e-5
    warmup: int = 200
    weight_decay: float = 0.01
    steps: int = 10000
    grad_clip: float = 1.0


@dataclass
class Frame:
    tokens: list[int]
    places: list[int]
    supervised: list[int]
    prompt_len: int
    frame_kind: str
    target: int
    metadata: dict[str, Any]


class Writer:
    def __init__(self):
        self.tokens: list[int] = []
        self.places: list[int] = []
        self.mask: list[int] = []

    def emit(self, token: int, place: int = 0, train: bool = False) -> None:
        self.tokens.append(int(token))
        self.places.append(int(place))
        self.mask.append(int(bool(train)))

    def num(self, n: int, *, origin: int = 1, width: Optional[int] = None,
            order: str = "lsd", train: bool = False) -> None:
        ds = num_digits(n, order=order, width=width)
        for i, tok in enumerate(ds):
            place = origin + (i if order == "lsd" else len(ds) - i - 1)
            self.emit(tok, place, train)

    def scalar(self, n: int, *, train: bool = False) -> None:
        for tok in num_digits(n, "msd"):
            self.emit(tok, 0, train)

    def into_frame(self, prompt_len: int, kind: str, target: int, **meta: Any) -> Frame:
        return Frame(self.tokens, self.places, self.mask, prompt_len, kind, target, meta)


def operand_prompt(a: int, b: int, *, order: str = "lsd") -> Writer:
    w = Writer()
    w.emit(BOS)
    w.num(a, order=order)
    w.emit(STAR)
    w.num(b, order=order)
    w.emit(EQUAL)
    return w


def make_flat(a: int, b: int, order: str = "msd") -> Frame:
    """A*B= prompt; predict exactly 2n digits followed by EOS."""
    w = operand_prompt(a, b, order="msd")
    p = len(w.tokens)
    n = max(digits_len(a), digits_len(b))
    w.num(a*b, width=2*n, order=order, train=True)
    w.emit(EOS, train=True)
    return w.into_frame(p, "flat_"+order, a*b, n=n, a=a, b=b)


def make_base(a: int, b: int, cutoff: int = 2) -> Frame:
    """RAT BASE scratchpad."""
    n = max(digits_len(a), digits_len(b))
    if n > cutoff:
        raise ValueError("BASE supervision is limited to cutoff-length operands")
    w = operand_prompt(a,b)
    p = len(w.tokens)
    for j in range(n):
        if j:
            w.emit(PLUS, train=True)
        bj = (b // 10**j) % 10
        w.num(a*bj, origin=1+j, width=n+1, train=True)
    w.emit(EQUAL, train=True)
    w.num(a*b, origin=1, width=2*n, train=True)
    w.emit(EOS, train=True)
    return w.into_frame(p, "base", a*b, a=a, b=b)


def make_split(a: int, b: int) -> Frame:
    """Learned router SPLIT, the k scalar and ordered four child pairs."""
    k, children = split_parts(a, b)
    w = operand_prompt(a,b)
    p = len(w.tokens)
    w.emit(SPLIT, train=True)
    w.scalar(k, train=True)
    for i, (u,v) in enumerate(children, 1):
        w.emit(CALL, train=True)
        w.emit(DIGIT_BASE+i, train=True)
        w.num(u, train=True)
        w.emit(BAR, train=True)
        w.num(v, train=True)
    w.emit(EOS, train=True)
    return w.into_frame(p, "split", a*b, a=a, b=b, k=k, children=children)


def combine_prompt(parts: Sequence[int], k: int) -> Writer:
    w=Writer()
    w.emit(BOS)
    w.emit(COMBINE)
    w.scalar(k)
    w.emit(BAR)
    for i,(v,shift) in enumerate(zip(parts, [0,k,k,2*k])):
        if i:
            w.emit(BAR)
        w.num(v,origin=1+shift)
    w.emit(EQUAL)
    return w


def make_combine(parts: Sequence[int], k: int) -> Frame:
    w=combine_prompt(parts,k)
    p=len(w.tokens)
    total=combine_numeric(parts,k)
    w.num(total,train=True)
    w.emit(EOS,train=True)
    return w.into_frame(p,"combine",total,k=k,parts=list(parts))


def legal_combine_case(rng: random.Random, k: int) -> tuple[list[int],int]:
    """Four products obtained from four legal k-digit operand halves."""
    al=rng.randrange(0,10**k); ah=rng.randrange(0,10**k)
    bl=rng.randrange(0,10**k); bh=rng.randrange(0,10**k)
    return [al*bl,al*bh,ah*bl,ah*bh],k


def randomize_frame_origin(frame:Frame,rng:random.Random,cfg:Config)->Frame:
    """Sample a legal local origin during training only."""
    peak=max(frame.places)
    if peak>cfg.place_max:
        raise ValueError("Frame place extent exceeds configured place_max")
    delta=rng.randrange(0,cfg.place_max-peak+1)
    frame.places=[p+delta if p>0 else 0 for p in frame.places]
    frame.metadata["origin"]=1+delta
    return frame


def draw_rat_frame(rng: random.Random, cfg: Config, *, forced_kind: Optional[str] = None,
                   combine_scale: int = 1) -> Frame:
    kind=forced_kind
    if kind is None:
        kind=rng.choices(["base","split","combine"],[0.15,0.10,0.75])[0]
    if kind=="base":
        n=rng.randrange(1,cfg.base_cutoff+1)
        return randomize_frame_origin(make_base(sample_number(rng,n),sample_number(rng,n),cfg.base_cutoff),rng,cfg)
    if kind=="split":
        n=rng.randrange(cfg.base_cutoff+1,cfg.split_copy_max+1)
        if rng.random()<0.85:
            n=rng.randrange(cfg.base_cutoff+1,cfg.split_train_max+1)
        return randomize_frame_origin(make_split(sample_number(rng,n),sample_number(rng,n)),rng,cfg)
    if kind=="combine":
        k_base=rng.randrange(1,cfg.combine_k_max+1)
        k=k_base*combine_scale
        partials,k=legal_combine_case(rng,k)
        return randomize_frame_origin(make_combine(partials,k),rng,cfg)
    raise ValueError(f"Unknown frame kind {kind}")


def make_batch(frames: Sequence[Frame], cfg: Config, device: torch.device) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    max_len=max(len(f.tokens) for f in frames)
    if max_len>cfg.max_seq:
        raise ValueError(f"frame length {max_len} exceeds max_seq={cfg.max_seq}; increase --max-seq")
    if max(max(f.places) for f in frames)>cfg.place_max:
        raise ValueError("place ID beyond configured place_max")
    toks=torch.full((len(frames),max_len),PAD,dtype=torch.long)
    places=torch.zeros((len(frames),max_len),dtype=torch.long)
    mask=torch.zeros((len(frames),max_len),dtype=torch.float32)
    for i,fr in enumerate(frames):
        t=len(fr.tokens)
        toks[i,:t]=torch.tensor(fr.tokens,dtype=torch.long)
        places[i,:t]=torch.tensor(fr.places,dtype=torch.long)
        mask[i,:t]=torch.tensor(fr.supervised,dtype=torch.float32)
    return toks.to(device),places.to(device),mask.to(device)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.d_model % cfg.heads:
            raise ValueError("d_model must be divisible by heads")
        self.h=cfg.heads
        self.d=cfg.d_model//cfg.heads
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model)
        self.proj=nn.Linear(cfg.d_model,cfg.d_model)
        self.dropout=cfg.dropout

    def forward(self,x:torch.Tensor)->torch.Tensor:
        b,t,c=x.shape
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        q=q.view(b,t,self.h,self.d).transpose(1,2)
        k=k.view(b,t,self.h,self.d).transpose(1,2)
        v=v.view(b,t,self.h,self.d).transpose(1,2)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=self.dropout if self.training else 0.0,is_causal=True)
        return self.proj(y.transpose(1,2).contiguous().view(b,t,c))


class Block(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        self.ln1=nn.LayerNorm(cfg.d_model)
        self.attn=CausalSelfAttention(cfg)
        self.ln2=nn.LayerNorm(cfg.d_model)
        self.ff=nn.Sequential(nn.Linear(cfg.d_model,cfg.d_ff),nn.GELU(),nn.Linear(cfg.d_ff,cfg.d_model))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=x+self.attn(self.ln1(x))
        x=x+self.ff(self.ln2(x))
        return x


class RATDecoder(nn.Module):
    def __init__(self,cfg:Config):
        super().__init__()
        self.cfg=cfg
        self.embed=nn.Embedding(len(TOKENS),cfg.d_model)
        self.place_embed=nn.Embedding(cfg.place_max+1,cfg.d_model)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln_f=nn.LayerNorm(cfg.d_model)
        self.head=nn.Linear(cfg.d_model,len(TOKENS),bias=False)
        self.head.weight=self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m:nn.Module)->None:
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,mean=0,std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,mean=0,std=0.02)

    def forward(self,tok:torch.Tensor,place:torch.Tensor)->torch.Tensor:
        if tok.shape!=place.shape:
            raise ValueError("tokens and places must match")
        if tok.size(1)>self.cfg.max_seq:
            raise ValueError("Sequence length exceeds max_seq")
        x=self.embed(tok)+self.place_embed(place)
        for block in self.blocks:
            x=block(x)
        return self.head(self.ln_f(x))

    def loss(self,tok:torch.Tensor,place:torch.Tensor,supervised:torch.Tensor)->torch.Tensor:
        logits=self(tok,place)
        if tok.shape[1]<2:
            raise ValueError("Training sequence too short")
        per=F.cross_entropy(logits[:,:-1].reshape(-1,logits.size(-1)),tok[:,1:].reshape(-1),reduction="none")
        m=supervised[:,1:].reshape(-1)
        return (per*m).sum()/m.sum().clamp_min(1.0)


def seed_everything(seed:int,deterministic:bool=False)->random.Random:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True,warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic=True
            torch.backends.cudnn.benchmark=False
    return random.Random(seed)


def resolve_device(value:str)->torch.device:
    if value!="auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path:str,model:RATDecoder,cfg:Config,seed:int,step:int,mode:str,extra:Optional[dict]=None,optimizer:Optional[torch.optim.Optimizer]=None)->None:
    out=Path(path)
    out.parent.mkdir(parents=True,exist_ok=True)
    record={"model":model.state_dict(),"config":asdict(cfg),"tokens":TOKENS,
            "seed":seed,"step":step,"mode":mode,"extra":extra or {}}
    if optimizer is not None:
        record["optimizer"]=optimizer.state_dict()
    torch.save(record,out)


def load_checkpoint(path:str,device:torch.device)->tuple[RATDecoder,dict]:
    data=torch.load(path,map_location="cpu",weights_only=False)
    if data.get("tokens")!=TOKENS:
        raise ValueError("Checkpoint tokenizer mismatch")
    cfg=Config(**data["config"])
    model=RATDecoder(cfg)
    model.load_state_dict(data["model"],strict=True)
    model.to(device).eval()
    return model,data


def train_model(cfg:Config, *, seed:int,output:str,mode:str="rat",order:str="msd",
                from_checkpoint:Optional[str]=None,forced_kind:Optional[str]=None,
                combine_scale:int=1,device:str="auto",deterministic:bool=False,
                log_interval:int=100, save_every:int=0, steps:Optional[int]=None,
                flat_max_digits:int=6)->str:
    rng=seed_everything(seed,deterministic)
    dev=resolve_device(device)
    if from_checkpoint:
        model,old=load_checkpoint(from_checkpoint,dev)
        cfg=model.cfg
    else:
        model=RATDecoder(cfg).to(dev)
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    offset=0
    if from_checkpoint:
        offset=int(old.get("step",0))
        if "optimizer" in old:
            optimizer.load_state_dict(old["optimizer"])
        else:
            print("WARNING: optimizer state missing; adaptation will reset AdamW moments",flush=True)
    final_steps=int(steps if steps is not None else cfg.steps)
    if final_steps<1:
        raise ValueError("Number of steps must be positive")
    for step in range(1,final_steps+1):
        current_lr=cfg.lr*min(1.,(offset+step)/max(1,cfg.warmup))
        for param in optimizer.param_groups:
            param["lr"]=current_lr
        examples: list[Frame]=[]
        attempts=0
        while len(examples)<cfg.batch_size:
            attempts+=1
            if attempts>1000*cfg.batch_size:
                raise RuntimeError("Unable to sample frames fitting max_seq/place_max")
            if mode=="flat":
                n=rng.randrange(1,flat_max_digits+1)
                fr=randomize_frame_origin(make_flat(sample_number(rng,n),sample_number(rng,n),order=order),rng,cfg)
            else:
                fr=draw_rat_frame(rng,cfg,forced_kind=forced_kind,combine_scale=combine_scale)
            if len(fr.tokens)<=cfg.max_seq and max(fr.places)<=cfg.place_max:
                examples.append(fr)
        tok,place,lossmask=make_batch(examples,cfg,dev)
        optimizer.zero_grad(set_to_none=True)
        loss=model.loss(tok,place,lossmask)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
        optimizer.step()
        if step==1 or step%log_interval==0 or step==final_steps:
            print(json.dumps({"step":step,"loss":round(loss.item(),6),"lr":current_lr,
                              "mode":mode,"device":str(dev)},sort_keys=True),flush=True)
        if save_every and step%save_every==0:
            save_checkpoint(str(Path(output).with_suffix(f".step{step}.pt")),model,cfg,seed,offset+step,mode,
                            {"order":order,"from_checkpoint":from_checkpoint,"combine_scale":combine_scale},optimizer)
    save_checkpoint(output,model,cfg,seed,offset+final_steps,mode,
                    {"order":order,"from_checkpoint":from_checkpoint,"combine_scale":combine_scale,
                     "flat_max_digits":flat_max_digits,"adaptation_updates":final_steps},optimizer)
    return output


@torch.no_grad()
def greedy_generate(model:RATDecoder,prompt:Writer,*,max_new_tokens:int=256,
                    device:Optional[torch.device]=None,forced_eos:bool=False)->list[int]:
    """Greedy autoregressive decoder; output includes EOS only if actually sampled."""
    device=device or next(model.parameters()).device
    model.eval()
    tok=prompt.tokens.copy()
    place=prompt.places.copy()
    generated=[]
    for _ in range(max_new_tokens):
        if len(tok)>=model.cfg.max_seq:
            break
        t=torch.tensor(tok,device=device,dtype=torch.long)[None,:]
        p=torch.tensor(place,device=device,dtype=torch.long)[None,:]
        nxt=int(model(t,p)[0,-1].argmax().item())
        generated.append(nxt)
        if nxt==EOS:
            break
        tok.append(nxt)
        place.append(infer_next_place(prompt,tok,generated))
    return generated


def infer_next_place(prompt:Writer,tokens:list[int],generated:list[int])->int:
    """Position ID assigned to latest generated token for use on the next step.

    The model is conditioned on a continuation's exact decimal lane. Since place
    IDs are side information, each generated digit gets its output lane by frame.
    For the learned SPLIT string we track CALL/field separators. Numeric fields
    are distinguished by emitted control tokens.
    """
    if not generated:
        return 0
    last=generated[-1]
    if last not in DIGIT_IDS:
        return 0
    if len(prompt.tokens)>=2 and prompt.tokens[1]==COMBINE:
        return sum(t in DIGIT_IDS for t in generated)
    if generated[0]==SPLIT:
        after_last_call=-1
        for i,t in enumerate(generated):
            if t==CALL:
                after_last_call=i
        if after_last_call<0 or len(generated)==after_last_call+2:
            return 0
        suffix=generated[after_last_call+2:]
        if BAR in suffix:
            suffix=suffix[suffix.index(BAR)+1:]
        return sum(t in DIGIT_IDS for t in suffix)
    sequence=generated
    if EQUAL in sequence:
        end=sequence[sequence.index(EQUAL)+1:]
        return sum(t in DIGIT_IDS for t in end)
    pos=sum(t==PLUS for t in sequence)
    suffix=sequence[sequence.index(PLUS)+1:] if PLUS in sequence else sequence
    return 1+pos+max(0,sum(t in DIGIT_IDS for t in suffix)-1)


def parse_base_continuation(tokens:Sequence[int])->Optional[int]:
    if not tokens or tokens[-1]!=EOS:
        return None
    pre=list(tokens[:-1])
    if EQUAL not in pre:
        return None
    i=len(pre)-1-pre[::-1].index(EQUAL)
    return parse_digits(pre[i+1:],"lsd")


def parse_split_continuation(tokens:Sequence[int])->Optional[tuple[int,list[tuple[int,int]]]]:
    if not tokens or tokens[0]!=SPLIT or tokens[-1]!=EOS:
        return None
    t=list(tokens)
    try:
        j=1
        while j<len(t) and t[j] in DIGIT_IDS:
            j+=1
        k=parse_digits(t[1:j],"msd")
        if k is None or k<=0:
            return None
        parts=[]
        for i in range(1,5):
            if t[j:j+2]!=[CALL,DIGIT_BASE+i]:
                return None
            j+=2
            start=j
            while j<len(t) and t[j] in DIGIT_IDS:
                j+=1
            u=parse_digits(t[start:j])
            if j>=len(t) or t[j]!=BAR:
                return None
            j+=1
            start=j
            while j<len(t) and t[j] in DIGIT_IDS:
                j+=1
            v=parse_digits(t[start:j])
            if u is None or v is None:
                return None
            parts.append((u,v))
        if j!=len(t)-1:
            return None
        return k,parts
    except (ValueError,IndexError):
        return None


def parse_combine_continuation(tokens:Sequence[int])->Optional[int]:
    if not tokens or tokens[-1]!=EOS:
        return None
    return parse_digits(tokens[:-1],"lsd")


class Executor:
    """Neural RAT recursion with optional exact oracles and trace logging.

    oracle_root_split: root SPLIT is exact, other SPLITs neural.
    oracle_all_split: all SPLITs exact, BASE/COMBINE neural.
    oracle_all_combine: COMBINE operations exact, others neural.
    oracle_all: all frames exact; tests deterministic arithmetic executor.
    """
    def __init__(self,model:Optional[RATDecoder],*,oracle_root_split:bool=False,
                 oracle_all_split:bool=False,oracle_all_combine:bool=False,
                 oracle_all:bool=False,max_depth:int=12,max_calls:int=2000,
                 max_new_tokens:int=350):
        self.model=model
        self.oracle_root_split=oracle_root_split
        self.oracle_all_split=oracle_all_split
        self.oracle_all_combine=oracle_all_combine
        self.oracle_all=oracle_all
        self.max_depth=max_depth
        self.max_calls=max_calls
        self.max_new_tokens=max_new_tokens
        self.trace:list[dict[str,Any]]=[]
        self.calls=0
        self.first_failure:Optional[str]=None

    def _record(self,frame:str,depth:int,oracle:bool,valid:bool,correct:bool,
                expected:Any,observed:Any,reason:str="") -> None:
        event={"call":self.calls,"frame":frame,"depth":depth,"oracle":oracle,
               "valid":valid,"correct":correct,"expected":expected,"observed":observed,"reason":reason}
        self.trace.append(event)
        if not correct and self.first_failure is None:
            self.first_failure="malformed" if not valid else frame

    def _generate(self,w:Writer)->list[int]:
        if self.model is None:
            raise ValueError("A model checkpoint is required for neural execution")
        self.calls+=1
        if self.calls>self.max_calls:
            raise RuntimeError("Maximum model calls exceeded")
        return greedy_generate(self.model,w,max_new_tokens=self.max_new_tokens)

    def multiply(self,a:int,b:int)->Optional[int]:
        self.trace=[]
        self.first_failure=None
        self.calls=0
        try:
            return self._multiply(a,b,0)
        except (RuntimeError,RecursionError) as exc:
            self._record("execution",0,False,False,False,a*b,None,str(exc))
            return None

    def _multiply(self,a:int,b:int,depth:int)->Optional[int]:
        n=max(digits_len(a),digits_len(b))
        if depth>self.max_depth:
            raise RuntimeError("Depth limit exceeded")
        if self.oracle_all:
            if n<=2:
                self._record("base",depth,True,True,True,a*b,a*b)
                return a*b
            k,children=split_parts(a,b)
            self._record("split",depth,True,True,True,(k,children),(k,children))
            partials=[]
            for u,v in children:
                child=self._multiply(u,v,depth+1)
                if child is None:
                    return None
                partials.append(child)
            value=combine_numeric(partials,k)
            self._record("combine",depth,True,True,value==a*b,a*b,value)
            return value
        if n<=2 and self.oracle_all_split:
            pass
        oracle_split=(n>2 and (self.oracle_all_split or (depth==0 and self.oracle_root_split)))
        if oracle_split:
            k,children=split_parts(a,b)
            self._record("split",depth,True,True,True,[k,split_parts(a,b)[1]],[k,children])
        else:
            tokens=self._generate(operand_prompt(a,b))
            if not tokens:
                self._record("split" if n>2 else "base",depth,False,False,False,a*b,None,"empty continuation")
                return None
            if tokens[0]!=SPLIT:
                value=parse_base_continuation(tokens)
                correct=value==a*b
                self._record("base",depth,False,value is not None,correct,a*b,value)
                return value
            parsed=parse_split_continuation(tokens)
            if parsed is None:
                self._record("split",depth,False,False,False,split_parts(a,b) if n>1 else None,None,"malformed SPLIT")
                return None
            k,children=parsed
            correct=(n>1 and (k,children)==split_parts(a,b))
            self._record("split",depth,False,True,correct,
                         split_parts(a,b) if n>1 else None,(k,children))
            if n<=2 or k>=n or any(max(digits_len(u),digits_len(v))>=n for u,v in children):
                self._record("split",depth,False,False,False,"strictly smaller children",children,"non-decreasing recursion")
                return None
        parts=[]
        for u,v in children:
            child=self._multiply(u,v,depth+1)
            if child is None:
                return None
            parts.append(child)
        expected=a*b
        if self.oracle_all_combine:
            value=combine_numeric(parts,k)
            self._record("combine",depth,True,True,value==expected,expected,value)
            return value
        tokens=self._generate(combine_prompt(parts,k))
        value=parse_combine_continuation(tokens)
        self._record("combine",depth,False,value is not None,value==expected,expected,value)
        return value


def test_pairs(n:int,count:int,seed:int)->list[tuple[int,int]]:
    rng=random.Random(seed+10007*n)
    return [(sample_number(rng,n),sample_number(rng,n)) for _ in range(count)]


def evaluate_rat(model:RATDecoder, *,lengths:Sequence[int],examples:int,seed:int,
                 oracle_root_split:bool=False,oracle_all_split:bool=False,
                 oracle_all_combine:bool=False,max_new_tokens:int=350,
                 trace_path:Optional[str]=None)->list[dict[str,Any]]:
    executor=Executor(model,oracle_root_split=oracle_root_split,oracle_all_split=oracle_all_split,
                      oracle_all_combine=oracle_all_combine,max_new_tokens=max_new_tokens)
    results=[]
    tf=None
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True,exist_ok=True)
        tf=open(trace_path,"w",encoding="utf-8")
    try:
        for n in lengths:
            successes=0; malformed=0; call_total=0
            for ix,(a,b) in enumerate(test_pairs(n,examples,seed)):
                pred=executor.multiply(a,b)
                success=pred==a*b
                successes+=int(success)
                malformed+=int(pred is None)
                call_total+=executor.calls
                if tf:
                    tf.write(json.dumps({"n":n,"example_id":ix,"a":a,"b":b,"truth":a*b,
                                         "prediction":pred,"correct":success,"calls":executor.calls,
                                         "first_failure":executor.first_failure,"trace":executor.trace})+"\n")
            row={"digits":n,"examples":examples,"correct":successes,
                 "accuracy_pct":round(successes*100/examples,4),
                 "malformed":malformed,"malformed_pct":round(malformed*100/examples,4),
                 "avg_model_calls":round(call_total/examples,3)}
            results.append(row)
            print(json.dumps(row,sort_keys=True),flush=True)
    finally:
        if tf:
            tf.close()
    return results


def write_csv(path:str,rows:Sequence[dict[str,Any]])->None:
    if not rows:
        return
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def self_test()->None:
    cfg=Config(layers=1,d_model=32,heads=4,d_ff=64,max_seq=512,batch_size=2)
    examples=[make_flat(12,34),make_flat(12,34,"lsd"),make_base(34,78),make_split(1234,5678),
              make_combine([2652,1904,936,672],2)]
    for fr in examples:
        assert len(fr.tokens)==len(fr.places)==len(fr.supervised)
        assert sum(fr.supervised)>0
    sp=make_split(1234,5678)
    assert parse_split_continuation(sp.tokens[sp.prompt_len:]) == split_parts(1234,5678)
    ba=make_base(34,78)
    assert parse_base_continuation(ba.tokens[ba.prompt_len:]) == 34*78
    co=make_combine([2652,1904,936,672],2)
    assert parse_combine_continuation(co.tokens[co.prompt_len:])==1234*5678
    assert combine_numeric([2652,1904,936,672],2)==7006652
    exec_=Executor(None,oracle_all=True)
    for a,b in [(0,0),(34,78),(1234,5678),(99999999,88888888)]:
        assert exec_.multiply(a,b)==a*b
    assert len(exec_.trace)==26, "An 8x8 product requires 16 BASE, 5 SPLIT and 5 COMBINE frames"
    model=RATDecoder(cfg)
    tok,place,mask=make_batch([make_base(12,34),make_base(56,78)],cfg,torch.device("cpu"))
    loss=model.loss(tok,place,mask)
    assert bool(torch.isfinite(loss))
    loss.backward()
    print("PASS: RAT frame serialization, exact executor, forward/backprop",flush=True)


def main(argv:Optional[Sequence[str]]=None)->None:
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=ap.add_subparsers(dest="command",required=True)
    t=sub.add_parser("train",help="Train RAT decoder on online BASE/SPLIT/COMBINE frames")
    t.add_argument("--out",required=True)
    t.add_argument("--seed",type=int,default=42)
    t.add_argument("--steps",type=int,default=10000)
    t.add_argument("--batch-size",type=int,default=64)
    t.add_argument("--layers",type=int,default=6)
    t.add_argument("--d-model",type=int,default=256)
    t.add_argument("--heads",type=int,default=8)
    t.add_argument("--d-ff",type=int,default=1024)
    t.add_argument("--max-seq",type=int,default=1024)
    t.add_argument("--lr",type=float,default=8e-5)
    t.add_argument("--warmup",type=int,default=200)
    t.add_argument("--weight-decay",type=float,default=0.01)
    t.add_argument("--device",default="auto")
    t.add_argument("--deterministic",action="store_true")
    t.add_argument("--log-interval",type=int,default=100)
    t.add_argument("--save-every",type=int,default=0)
    p=sub.add_parser("predict",help="Predict using recursive model, or exact arithmetic oracle")
    p.add_argument("--checkpoint")
    p.add_argument("--a",type=int,required=True)
    p.add_argument("--b",type=int,required=True)
    p.add_argument("--device",default="auto")
    p.add_argument("--oracle-all",action="store_true")
    p.add_argument("--max-new-tokens",type=int,default=350)
    e=sub.add_parser("evaluate",help="Test exact match for operand lengths")
    e.add_argument("--checkpoint",required=True)
    e.add_argument("--n",type=int,nargs="+",default=[3,4,5,6,7,8,9,10])
    e.add_argument("--examples",type=int,default=1000)
    e.add_argument("--seed",type=int,default=20260816)
    e.add_argument("--device",default="auto")
    e.add_argument("--trace-path")
    e.add_argument("--out-csv")
    e.add_argument("--max-new-tokens",type=int,default=350)
    sub.add_parser("self-test")
    a=ap.parse_args(argv)
    if a.command=="self-test":
        return self_test()
    if a.command=="train":
        cfg=Config(layers=a.layers,d_model=a.d_model,heads=a.heads,d_ff=a.d_ff,
                   max_seq=a.max_seq,batch_size=a.batch_size,lr=a.lr,
                   warmup=a.warmup,weight_decay=a.weight_decay,steps=a.steps)
        train_model(cfg,seed=a.seed,output=a.out,device=a.device,
                    deterministic=a.deterministic,log_interval=a.log_interval,save_every=a.save_every)
    elif a.command=="predict":
        dev=resolve_device(a.device)
        model=None if a.oracle_all else load_checkpoint(a.checkpoint,dev)[0] if a.checkpoint else None
        result=Executor(model,oracle_all=a.oracle_all,max_new_tokens=a.max_new_tokens)
        pred=result.multiply(a.a,a.b)
        print(json.dumps({"a":a.a,"b":a.b,"pred":pred,"truth":a.a*a.b,
                          "correct":pred==a.a*a.b,"first_failure":result.first_failure,
                          "model_calls":result.calls,"trace":result.trace},indent=2))
    elif a.command=="evaluate":
        model,_=load_checkpoint(a.checkpoint,resolve_device(a.device))
        rows=evaluate_rat(model,lengths=a.n,examples=a.examples,seed=a.seed,
                          trace_path=a.trace_path,max_new_tokens=a.max_new_tokens)
        if a.out_csv:
            write_csv(a.out_csv,rows)


if __name__=="__main__":
    main()
