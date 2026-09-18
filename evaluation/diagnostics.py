"""Auditable evaluation primitives. No model imports or API calls at import time."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

METRICS = ('faithfulness','answer_relevancy','llm_context_precision_with_reference','context_recall')


def sample_id(sample):
    payload=json.dumps([sample.get('user_input',''),sample.get('reference','')],ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def read_replay_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        raw=row.get('retrieved_contexts','')
        try:
            contexts=json.loads(raw)
        except (ValueError,TypeError):
            try: contexts=ast.literal_eval(raw)
            except (ValueError,SyntaxError): contexts=None
        row['retrieved_contexts']=contexts
    return rows


def finite(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (TypeError,ValueError): return None


def summarize(rows, metrics=METRICS):
    result={}
    for metric in metrics:
        values=[v for row in rows if (v:=finite(row.get(metric))) is not None]
        n=len(rows)
        result[metric]=dict(total=n,valid=len(values),missing=n-len(values),
            missing_rate=(n-len(values))/n if n else None,
            mean=statistics.mean(values) if values else None,
            median=statistics.median(values) if values else None,
            sample_std=statistics.stdev(values) if len(values)>1 else None,
            statuses=dict(Counter(row.get(metric+'_status','ok' if finite(row.get(metric)) is not None else 'historical_missing_unknown') for row in rows)))
    return result


def input_error(sample,metric):
    if sample.get('generation_status')=='error': return 'generation_error'
    fields=['user_input']
    if metric in ('faithfulness','answer_relevancy'): fields.append('response')
    if metric in ('llm_context_precision_with_reference','context_recall'): fields.append('reference')
    if any(not isinstance(sample.get(f),str) or not sample[f].strip() for f in fields):
        return 'required_text_empty'
    if metric != 'answer_relevancy':
        ctx=sample.get('retrieved_contexts')
        if not isinstance(ctx,list) or not ctx or not all(isinstance(c,str) for c in ctx) or not any(c.strip() for c in ctx):
            return 'contexts_invalid_or_empty'
    return None


def score_samples(samples,metrics,scorer,*,attempts=1,audit_path):
    """scorer(sample,metric,attempt) -> float. Persist each attempt immediately.

    A NaN without exception remains non_finite_score (no invented root cause).
    Exception messages may contain provider data/secrets: save type, not raw text.
    """
    if attempts<1: raise ValueError('attempts must be positive')
    audit_path=Path(audit_path)
    audit_path.parent.mkdir(parents=True,exist_ok=True)
    output=[]
    with audit_path.open('x',encoding='utf-8') as audit:
        for index,sample in enumerate(samples,1):
            row={k:sample.get(k) for k in ('user_input','response','reference','retrieved_contexts','synthesizer_name','generation_status','generation_error_type')}
            row.update(sample_id=sample_id(sample),source_row=index)
            for metric in metrics:
                value=None
                err=input_error(sample,metric)
                for attempt in range(1,attempts+1):
                    start=time.monotonic()
                    status='ok'; error_type=None
                    if err: status='input_invalid'; error_type=err
                    else:
                        try:
                            value=finite(scorer(sample,metric,attempt))
                            if value is None: status='non_finite_score'
                        except Exception as exc:
                            status='evaluator_exception'; error_type=type(exc).__name__
                    event=dict(sample_id=row['sample_id'],source_row=index,metric=metric,attempt=attempt,
                               status=status,error_type=error_type,value=value,elapsed_seconds=time.monotonic()-start)
                    audit.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n'); audit.flush()
                    if status=='ok' or err: break
                row[metric]=value
                row[metric+'_status']='ok_after_retry' if status=='ok' and attempt>1 else status
                row[metric+'_attempts']=attempt
            output.append(row)
    return output
