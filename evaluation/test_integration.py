"""
evaluation/test_integration.py
================================
集成测试。在跑 RAGAS 完整评估（几美元）之前，先用 3 个样本验证：
  ✓ rag_adapter 能调通 src/main.py
  ✓ query() 返回的 (answer, contexts) 格式正确
  ✓ contexts 是中文法律文本，不是空列表也不是乱码

跑通这个脚本后再跑 run_evaluation.py。

运行：
    cd legal-rag-system
    python evaluation/test_integration.py
"""

import json
import sys
from pathlib import Path

# 让 evaluation/ 目录可以 import 同级文件
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_adapter import get_rag_system  # noqa: E402


TESTSET_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "testset.json"
N_SAMPLES = 3  # 只跑前 3 个，快速验证


def main():
    print("=" * 70)
    print("集成测试：rag_adapter ←→ src/main.py")
    print("=" * 70)

    # 1. 加载测试集前 N 个问题
    if not TESTSET_PATH.exists():
        print(f"❌ 测试集不存在: {TESTSET_PATH}")
        print("   请先把 testset.json 移动到 data/eval/ 下")
        sys.exit(1)

    samples = json.loads(TESTSET_PATH.read_text(encoding="utf-8"))[:N_SAMPLES]
    print(f"\n>>> 加载 {len(samples)} 个测试样本")

    # 2. 实例化 baseline
    print(f"\n>>> 实例化 BaselineRAG（首次会构建索引，约 30 秒）...")
    rag = get_rag_system("baseline")
    print(f"    ✓ 实例化成功")

    # 3. 跑测试
    print(f"\n>>> 开始测试...\n")
    all_pass = True
    for i, sample in enumerate(samples, 1):
        question = sample["question"]
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"[样本 {i}/{len(samples)}]")
        print(f"问题: {question}")

        try:
            answer, contexts = rag.query(question)
        except Exception as e:
            print(f"❌ query() 抛异常: {e}")
            all_pass = False
            continue

        # 检查格式
        checks = [
            ("answer 是字符串", isinstance(answer, str)),
            ("answer 非空", bool(answer and answer.strip())),
            ("contexts 是 list", isinstance(contexts, list)),
            ("contexts 非空", len(contexts) > 0),
            ("contexts 元素是字符串", all(isinstance(c, str) for c in contexts)),
        ]
        print()
        for desc, ok in checks:
            mark = "✓" if ok else "❌"
            print(f"  {mark} {desc}")
            if not ok:
                all_pass = False

        # 预览输出
        print(f"\n  [答案预览] {answer[:200]}{'...' if len(answer) > 200 else ''}")
        print(f"\n  [Contexts] 共 {len(contexts)} 个，第一个预览:")
        if contexts:
            print(f"    {contexts[0][:200]}{'...' if len(contexts[0]) > 200 else ''}")
        print()

    # 4. 总结
    print("=" * 70)
    if all_pass:
        print("✅ 集成测试通过！现在可以放心跑完整评估：")
        print("   python evaluation/run_evaluation.py --strategy baseline")
    else:
        print("❌ 集成测试发现问题，请先修复再跑评估")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
