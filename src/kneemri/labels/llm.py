"""Grades multilingual radiology reports with an open instruction-tuned LLM (Qwen2.5-14B-Instruct, 4-bit).

Each report gets a 0-3 grade for each of the 12 findings, returned as JSON. Besides the hard grade, the
probabilities of the tokens "0".."3" are recorded at every generated grade digit, and the **expected grade**
E = sum(k * p_k) is kept. Reports with the same hard grade then get different scores, which matters for AUC.

Run one worker per GPU (``shard`` / ``n_shards``); on Kaggle 2x T4 a 4-bit 14B model fits once per GPU.
"""
import re
import time

import numpy as np
import pandas as pd
import torch

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


class ReportGrader:
    def __init__(self, model_id="Qwen/Qwen2.5-14B-Instruct", quant_4bit=True, max_chars=3000, max_new_tokens=120):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        kw = dict(device_map={"": 0}) if torch.cuda.is_available() else {}
        if quant_4bit:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                           bnb_4bit_compute_dtype=torch.float16,
                                                           bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        self.device = next(self.model.parameters()).device
        self.digits = [self.tok.convert_tokens_to_ids(str(k)) for k in range(4)]
        self.max_chars, self.max_new_tokens = max_chars, max_new_tokens

    def prompt(self, report):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": INSTRUCTION.format(report=str(report)[:self.max_chars])}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _generate(self, prompts):
        from transformers import LogitsProcessor, LogitsProcessorList
        digits = self.digits

        class DigitRecorder(LogitsProcessor):
            # log-probabilities of "0".."3" at every generation step (4 numbers per step)
            def __init__(self):
                self.steps = []

            def __call__(self, input_ids, scores):
                self.steps.append(torch.log_softmax(scores.float(), -1)[:, digits].cpu())
                return scores

        try:
            enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
            rec = DigitRecorder()
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.eos_token_id,
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
                if int(gen_ids[b, t]) not in digits:
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
