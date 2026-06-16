import sys
sys.path.insert(0, '/Users/zhangxinyu/sunxidan-06/0616-e/sxd-1189-1')

import pandas as pd
from app import parse_date_smart, detect_encoding, DATE_FORMATS

print("=" * 60)
print("🔍 测试 1：混合日期格式解析验证")
print("=" * 60)

test_dates = [
    '2026-06-02',
    '2026/06/02',
    '06/02/2026',
    '2026年06月02日',
    '02-06-2026',
    '20260602',
    '2026.06.02',
    '2026-06-02 09:15:00',
    '06/02/26',
    '06/03/2026 10:30 AM',
    '1717287300',
]

print(f"\n测试 {len(test_dates)} 种不同格式的日期：")
result = parse_date_smart(pd.Series(test_dates))
all_ok = True
for i, (original, parsed) in enumerate(zip(test_dates, result)):
    status = "✅" if pd.notna(parsed) else "❌"
    if pd.notna(parsed):
        parsed_str = parsed.strftime('%Y-%m-%d %H:%M:%S')
        print(f"  {status} {original:<25} -> {parsed_str}")
    else:
        print(f"  {status} {original:<25} -> 解析失败！")
        all_ok = False

print(f"\n解析成功率：{result.notna().sum()}/{len(test_dates)}")

print("\n" + "=" * 60)
print("🔍 测试 1b：智能格式检测（根据数据统计调整）")
print("=" * 60)

print("\n测试智能检测 MM/DD/YYYY 格式占优的数据：")
mmdd_dates = ['06/02/2026', '06/03/2026', '06/04/2026', '06/05/2026', '05/31/2026']
result_mmdd = parse_date_smart(pd.Series(mmdd_dates))
for orig, parsed in zip(mmdd_dates, result_mmdd):
    print(f"  {orig:<15} -> {parsed.strftime('%Y-%m-%d') if pd.notna(parsed) else '失败'}")

print("\n测试智能检测 DD/MM/YYYY 格式占优的数据：")
ddmm_dates = ['02/06/2026', '03/06/2026', '04/06/2026', '05/06/2026', '13/06/2026']
result_ddmm = parse_date_smart(pd.Series(ddmm_dates))
for orig, parsed in zip(ddmm_dates, result_ddmm):
    print(f"  {orig:<15} -> {parsed.strftime('%Y-%m-%d') if pd.notna(parsed) else '失败'}")

print("\n" + "=" * 60)
print("📂 测试 2：测试数据文件解析验证")
print("=" * 60)

import io
with open('/Users/zhangxinyu/sunxidan-06/0616-e/sxd-1189-1/test_mixed_dates.csv', 'rb') as f:
    content = f.read()
encoding = detect_encoding(content)
df = pd.read_csv(io.BytesIO(content), encoding=encoding)

print(f"\n文件编码：{encoding}")
print(f"总行数：{len(df)}")

parsed_dates = parse_date_smart(df['record_date'])
success_count = parsed_dates.notna().sum()
print(f"日期解析成功：{success_count}/{len(df)} 行")

failed = df[parsed_dates.isna()]['record_date']
if len(failed) > 0:
    print("\n❌ 解析失败的日期：")
    for idx, val in failed.items():
        print(f"  行 {idx+2}: {val}")
else:
    print("✅ 所有日期均解析成功！")

date_range = parsed_dates.min(), parsed_dates.max()
print(f"\n日期范围：{date_range[0].strftime('%Y-%m-%d')} ~ {date_range[1].strftime('%Y-%m-%d')}")
print(f"涉及天数：{(date_range[1] - date_range[0]).days + 1} 天")

print("\n" + "=" * 60)
print("📊 测试 3：饮品重做率验证（含小样本）")
print("=" * 60)

df['is_remake'] = df['remake_flag'].apply(lambda x: 1 if str(x).strip() in ['1', '是', 'True'] else 0)
drink_remake = df.groupby('drink_name').agg(
    订单数=('is_remake', 'count'),
    重做数=('is_remake', 'sum'),
    重做率=('is_remake', 'mean')
).reset_index()
drink_remake = drink_remake.sort_values('重做率', ascending=False)
drink_remake['样本量'] = drink_remake['订单数'].apply(lambda x: '⚠️ 小样本' if x < 3 else '✅ 可信')

print("\n所有饮品重做率：")
print("-" * 60)
print(f"{'饮品':<15} {'订单数':>6} {'重做数':>6} {'重做率':>8} {'可靠性':>10}")
print("-" * 60)
for _, row in drink_remake.iterrows():
    marker = "🔴" if row['样本量'] == '⚠️ 小样本' and row['重做率'] > 0 else "🟢"
    print(f"{marker} {row['drink_name']:<13} {row['订单数']:>6} {row['重做数']:>6} {row['重做率']*100:>7.1f}% {row['样本量']:>10}")

small_sample_high = drink_remake[(drink_remake['订单数'] < 3) & (drink_remake['重做率'] > 0)]
if len(small_sample_high) > 0:
    print(f"\n⚠️  发现 {len(small_sample_high)} 个小样本但高重做率的饮品（原逻辑会被隐藏）：")
    for _, row in small_sample_high.iterrows():
        print(f"   - {row['drink_name']}: {row['订单数']}单, 重做率{row['重做率']*100:.0f}%")
else:
    print("\n✅ 无小样本高重做率饮品")

print("\n" + "=" * 60)
print("📋 测试 4：HTML 报告内容验证")
print("=" * 60)

from app import generate_html_report, get_prep_suggestions

df['record_date_parsed'] = parsed_dates
df['brew_duration'] = 5.0
df['order_min_parsed'] = 500
anomaly_df = pd.DataFrame(columns=['行号', '类型', '原始值', '说明'])
summary_metrics = {
    "总订单数": f"{len(df)} 单",
    "覆盖门店": f"{df['store_name'].nunique()} 家",
    "平均出杯时长": "5.0 分钟",
    "整体重做率": f"{df['is_remake'].mean()*100:.1f}%"
}
prep_suggestions = get_prep_suggestions(df)

html = generate_html_report(df, df, anomaly_df, summary_metrics, prep_suggestions)

checks = {
    "核心指标": "核心指标" in html,
    "门店统计表": "门店统计" in html,
    "饮品统计表": "饮品统计" in html,
    "咖啡师统计表": "咖啡师统计" in html,
    "近14天趋势图": "近14天效率趋势" in html,
    "重做率Top10": "饮品重做率 Top 10" in html,
    "备料建议": "备料调整建议" in html,
    "异常明细": "异常明细" in html,
    "明细数据": "明细数据" in html,
    "CSS样式": "metric-card" in html,
    "打印样式": "@media print" in html,
}

print("\nHTML 报告内容检查：")
all_passed = True
for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"  {status} {check}")
    if not passed:
        all_passed = False

if all_passed:
    print("\n✅ 所有 HTML 报告模块检查通过！")
else:
    print("\n❌ 部分模块缺失，请检查！")

print("\n" + "=" * 60)
if all_ok and success_count == len(df) and all_passed:
    print("🎉 所有测试通过！修复生效。")
else:
    print("⚠️  部分测试未通过，请检查。")
print("=" * 60)
