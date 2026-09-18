"""Recompute historical coverage and the paired recall comparison without API calls."""
import argparse
import hashlib
import json
from pathlib import Path
from diagnostics import METRICS, finite, read_replay_csv, sample_id, summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'data/eval/results')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    data={};loaded={}
    for path in sorted(args.results.glob('*.csv')):
        if path.name.startswith('summary'): continue
        rows=read_replay_csv(path)
        if not rows or not all(m in rows[0] for m in METRICS): continue
        loaded[path.name]=rows
        data[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows),
            'unique_question_reference_ids':len({sample_id(r) for r in rows}),'metrics':summarize(rows),
            'empty_response_count':sum(not r.get('response','').strip() for r in rows),
            'missing_cells':[dict(source_row=i, sample_id=sample_id(r),metric=m,status='historical_missing_unknown') for i,r in enumerate(rows,1) for m in METRICS if finite(r.get(m)) is None]}
    a='hybrid_rerank_v1_20260501_181710.csv'; b='hybrid_rerank_v2_m3_20260501_222955.csv'
    pair=None
    if a in loaded and b in loaded:
        left={sample_id(r):r for r in loaded[a]};right={sample_id(r):r for r in loaded[b]}
        assert len(left)==len(loaded[a]) and len(right)==len(loaded[b]),'Duplicate paired IDs'
        assert left.keys()==right.keys(),'Question/reference sets differ'
        assert all(finite(r['context_recall']) is not None for r in list(left.values())+list(right.values()))
        n=len(left);diffs=[float(right[k]['context_recall'])-float(left[k]['context_recall']) for k in left]
        pair=dict(left=a,right=b,n=n,mean_left=sum(float(r['context_recall']) for r in left.values())/n,
                  mean_right=sum(float(r['context_recall']) for r in right.values())/n,mean_difference=sum(diffs)/n,
                  improved=sum(v>0 for v in diffs),unchanged=sum(v==0 for v in diffs),worsened=sum(v<0 for v in diffs))
    (args.out/'rag-history.json').write_text(json.dumps(dict(files=data,paired_recall=pair),ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# RAG 历史数据复算','', '本报告只重算保存的 CSV；未重新检索、生成或调用模型。历史错误原因未被保存，缺失统一标注 historical_missing_unknown。','', '| 文件 | 指标 | 有效/总数 | 缺失率 | 均值 |','| --- | --- | --- | --- | --- |']
    for name,record in data.items():
        for metric,stats in record['metrics'].items():
            mean=f"{stats['mean']:.6f}" if stats['mean'] is not None else 'N/A'
            lines.append(f"| {name} | {metric} | {stats['valid']}/{stats['total']} | {stats['missing_rate']:.1%} | {mean} |")
    if pair:
        lines+=['','## 同题、同参考答案的召回对比','',f"两组各 {pair['n']} 条且召回评分完整：{pair['mean_left']:.6f} → {pair['mean_right']:.6f}，差 {pair['mean_difference']:.6f}。",f"逐题：提高 {pair['improved']}、不变 {pair['unchanged']}、降低 {pair['worsened']}。",'', '仅是历史两种重排配置的观察对比；未冻结所有环境变量，不能单独归因于重排模型。48 题已用于迭代，是回归集。','', '不计算四指标平均的“综合分”；切题度每组仅 2–6 个有效值，不能用其标准差描述运行稳定性。']
    (args.out/'rag-history.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(pair,ensure_ascii=False))

if __name__=='__main__': main()
