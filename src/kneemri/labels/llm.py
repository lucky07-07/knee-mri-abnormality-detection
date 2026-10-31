"""Grades multilingual radiology reports with an open instruction-tuned LLM (Qwen2.5-14B-Instruct, 4-bit).

Each report gets a 0-3 grade for each of the 12 findings, returned as JSON. Besides the hard grade, the
probabilities of the tokens "0".."3" are recorded at every generated grade digit, and the **expected grade**
E = sum(k * p_k) is kept. Reports with the same hard grade then get different scores, which matters for AUC.

Run one worker per GPU (``shard`` / ``n_shards``); on Kaggle 2x T4 a 4-bit 14B model fits once per GPU.

The grader is model-agnostic (used for the LLM benchmark in notebook 06): the prompt and scoring are identical for
every model. Chat templates without a system role (Gemma) get the system text prepended to the user turn, and a
grade digit may be spelled by several vocabulary tokens (``"0"``, ``" 0"``, ...), whose probabilities are summed.
"""
import re
import time

import numpy as np
import pandas as pd

from kneemri.constants import LABELS, LLM_KEYS

SYSTEM = ("You are a musculoskeletal radiologist. You read knee MRI reports written in any language and grade "
          "findings. Answer with JSON only.")
INSTRUCTION = """Grade each finding for THIS knee from the report below (the report may be in any language; read it carefully, including negations).
Grades: 0 = absent, normal, or not mentioned; 1 = mild, small, equivocal, possible, or low-grade; 2 = definite or moderate; 3 = severe, large, complete, or high-grade.

Findings:
ACL  = anterior cruciate ligament injury (sprain, partial or complete tear, graft tear). Mucoid degeneration alone = 0.
MCL  = medial collateral ligament injury (sprain any grade, partial or complete tear).
MM   = medial meniscus TEAR (degeneration or grade 1-2 intrasubstance signal without tear = 0).
LM   = lateral meniscus TEAR (same rule).
MOA  = osteoarthritis of the medial tibiofemoral compartment (cartilage loss, chondral defects, osteophytes, joint space narrowing, subchondral edema or sclerosis).
LOA  = osteoarthritis of the lateral tibiofemoral compartment (same signs, lateral femoral condyle or lateral tibial plateau).
PFOA = patellofemoral osteoarthritis: chondromalacia or cartilage loss of the patella or trochlea, patellofemoral osteophytes.
EFF  = joint effusion (physiological trace fluid = 0, small = 1, moderate = 2, large = 3).
SYN  = synovitis, synovial thickening or hypertrophy, Hoffa fat pad edema/synovitis.
BAK  = Baker's (popliteal) cyst.
CON  = bone contusion / bone bruise / traumatic bone marrow edema.
FX   = fracture (including avulsion and subchondral insufficiency fracture).

Report:
\"\"\"{report}\"\"\"

Return ONE JSON object with an integer grade 0-3 for every key, for example:
{{"ACL":0,"MCL":0,"MM":2,"LM":0,"MOA":1,"LOA":0,"PFOA":1,"EFF":1,"SYN":0,"BAK":0,"CON":0,"FX":0}}"""


def parse_grades(text):
    out = []
    for k in LLM_KEYS:
        m = re.search(r'"%s"\s*:\s*"?(\d)' % k, text)
        out.append(min(3, int(m.group(1))) if m else None)
    return out


def grade_digit_ids(tok):
    """Vocabulary ids spelling each grade digit 0..3, with or without a leading space (``"0"``, ``" 0"``, ``"▁0"``).

    Tokenizers differ: Qwen and Gemma split digits into single tokens, Llama 3 also has space-prefixed digit tokens.
    The probability of grade k is the summed probability of all its spellings.
    """
    ids = {k: set() for k in range(4)}
    for i in range(len(tok)):
        s = tok.decode([i], skip_special_tokens=True).replace("▁", " ").strip()
        if s in ("0", "1", "2", "3"):
            ids[int(s)].add(i)
    missing = [k for k, v in ids.items() if not v]
    if missing:
        raise ValueError(f"tokenizer has no single-token spelling for grade(s) {missing}")
    return [sorted(ids[k]) for k in range(4)]


def chat_prompt(tok, system, user):
    """Chat-formatted prompt; templates without a system role (e.g. Gemma) get the system text in the user turn."""
    try:
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
    except Exception:
        return tok.apply_chat_template([{"role": "user", "content": f"{system}\n\n{user}"}],
                                       tokenize=False, add_generation_prompt=True)


class ReportGrader:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct", quant_4bit=True, max_chars=3000, max_new_tokens=120,
                 compute_dtype="float16", attn_implementation=None, token=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        compute_dtype = getattr(torch, compute_dtype) if isinstance(compute_dtype, str) else compute_dtype

        self.tok = AutoTokenizer.from_pretrained(model_id, token=token)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:                                       # Llama / Mistral ship without one
            self.tok.pad_token = self.tok.eos_token
        kw = dict(device_map={"": 0}) if torch.cuda.is_available() else {}
        if quant_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=compute_dtype,
                                                           bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = compute_dtype if torch.cuda.is_available() else torch.float32
        if attn_implementation:                                              # Gemma 2 wants "eager"
            kw["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=token, **kw).eval()
        self.device = next(self.model.parameters()).device
        self.digit_ids = grade_digit_ids(self.tok)
        self.digit_set = {i for ids in self.digit_ids for i in ids}
        self.max_chars, self.max_new_tokens = max_chars, max_new_tokens

    def prompt(self, report):
        return chat_prompt(self.tok, SYSTEM, INSTRUCTION.format(report=str(report)[:self.max_chars]))

    def _generate(self, prompts):
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList
        digit_ids = self.digit_ids

        class DigitRecorder(LogitsProcessor):
            # log-probability of each grade 0..3 at every generation step (summed over its token spellings)
            def __init__(self):
                self.steps = []

            def __call__(self, input_ids, scores):
                lp = torch.log_softmax(scores.float(), -1)
                self.steps.append(torch.stack([lp[:, ids].logsumexp(-1) for ids in digit_ids], -1).cpu())
                return scores

        try:
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            rec = DigitRecorder()
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id,
                                          logits_processor=LogitsProcessorList([rec]))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(prompts) == 1:
                return [([None] * 12, [np.nan] * 12)]
            h = len(prompts) // 2
            return self._generate(prompts[:h]) + self._generate(prompts[h:])
        gen_ids = out[:, enc["input_ids"].shape[1]:].cpu()
        texts = self.tok.batch_decode(gen_ids, skip_special_tokens=True)
        results = []
        for b, text in enumerate(texts):
            hard = parse_grades(text)
            expected = [np.nan if h is None else float(h) for h in hard]     # fallback per key: the hard grade
            for t in range(min(gen_ids.shape[1], len(rec.steps))):
                if int(gen_ids[b, t]) not in self.digit_set:
                    continue
                prefix = self.tok.decode(gen_ids[b, :t], skip_special_tokens=True)
                last, j = max((prefix.rfind('"%s"' % k), j) for j, k in enumerate(LLM_KEYS))
                if last < 0 or prefix[last:].count(":") != 1:                # digit is not a grade value
                    continue
                p = rec.steps[t][b].exp()
                expected[j] = float((p / p.sum() * torch.arange(4)).sum())
            results.append((hard, expected))
        return results

    def grade(self, reports, batch_size=16, token_budget=12000, log_every=10):
        """Returns a DataFrame with ``<finding>__hard`` and ``<finding>__exp`` columns, in input order."""
        prompts = [self.prompt(r) for r in reports]
        lens = np.array([len(x) for x in self.tok(prompts)["input_ids"]])
        batches, cur = [], []
        for i in np.argsort(lens):                                           # length-sorted, token-budgeted batches
            if cur and (len(cur) + 1 > batch_size or (len(cur) + 1) * lens[i] > token_budget):
                batches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        hard, exp = [None] * len(prompts), [None] * len(prompts)
        t0 = time.time()
        for bi, idx in enumerate(batches):
            for i, (h, e) in zip(idx, self._generate([prompts[i] for i in idx])):
                hard[i], exp[i] = h, e
            if bi % log_every == 0:
                print(f"{sum(len(b) for b in batches[:bi + 1])}/{len(prompts)} reports, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
        cols = [f"{c}__hard" for c in LABELS] + [f"{c}__exp" for c in LABELS]
        return pd.DataFrame([[np.nan if v is None else v for v in h] + list(e) for h, e in zip(hard, exp)],
                            columns=cols)
