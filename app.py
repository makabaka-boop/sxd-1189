import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import base64
from collections import defaultdict

st.set_page_config(
    page_title="咖啡门店出杯效率分析看板",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

REQUIRED_FIELDS = {
    "record_date": "记录日期",
    "store_name": "门店名称",
    "drink_name": "饮品名称",
    "order_minute": "下单时间(分钟数)",
    "finish_minute": "完成时间(分钟数)",
    "remake_flag": "重做标记",
    "barista_name": "咖啡师姓名",
    "note": "备注"
}

FIELD_SYNONYMS = {
    "record_date": ["record_date", "date", "日期", "记录日期", "下单日期", "order_date", "订单日期"],
    "store_name": ["store_name", "store", "门店", "门店名称", "店铺", "店铺名称"],
    "drink_name": ["drink_name", "drink", "饮品", "饮品名称", "饮料", "产品名称", "product", "product_name"],
    "order_minute": ["order_minute", "order_time", "下单时间", "下单时刻", "order_at"],
    "finish_minute": ["finish_minute", "finish_time", "完成时间", "完成时刻", "出杯时间", "finish_at"],
    "remake_flag": ["remake_flag", "remake", "重做", "重做标记", "是否重做", "is_remake"],
    "barista_name": ["barista_name", "barista", "咖啡师", "咖啡师姓名", "操作员", "operator"],
    "note": ["note", "备注", "说明", "comments", "remark"]
}

def detect_encoding(content_bytes):
    try:
        content_bytes.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        return 'gbk'

def parse_time_to_minute(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        if 0 <= value <= 1440:
            return float(value)
        return np.nan
    s = str(value).strip()
    if not s or s.lower() in ['nan', 'none', 'nat', '-']:
        return np.nan
    if s.isdigit():
        num = float(s)
        if 0 <= num <= 1440:
            return num
        return np.nan
    for sep in [':', '：', '.', '-']:
        if sep in s:
            parts = s.split(sep)
            try:
                h = int(parts[0])
                m = int(parts[1]) if len(parts) > 1 else 0
                if 0 <= h <= 23 and 0 <= m <= 59:
                    return h * 60 + m
            except ValueError:
                pass
    try:
        dt = pd.to_datetime(s, errors='raise')
        return dt.hour * 60 + dt.minute
    except (ValueError, TypeError):
        return np.nan

def parse_remake_flag(value):
    if pd.isna(value):
        return 0
    s = str(value).strip().lower()
    if s in ['1', 'true', 'yes', '是', 'y', '重做', '重制', '返工', '有', '有问题']:
        return 1
    if s in ['0', 'false', 'no', '否', 'n', '正常', '无', '没有']:
        return 0
    if any(keyword in s for keyword in ['重做', '重制', '返工', 'remake', 'redone']):
        return 1
    return 0

DATE_FORMATS = [
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%Y.%m.%d %H:%M:%S',
    '%Y-%m-%d %H:%M', '%Y/%m/%d %H:%M', '%Y.%m.%d %H:%M',
    '%Y-%m-%d %I:%M %p', '%Y/%m/%d %I:%M %p',
    '%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d',
    '%Y%m%d',
    '%Y年%m月%d日', '%Y年%m月%d', '%y年%m月%d日',
    '%m/%d/%Y %H:%M:%S', '%m-%d-%Y %H:%M:%S',
    '%m/%d/%Y %H:%M', '%m-%d-%Y %H:%M',
    '%m/%d/%Y', '%m-%d-%Y',
    '%m/%d/%y', '%m-%d-%y',
    '%d/%m/%Y', '%d-%m-%Y',
    '%d/%m/%y', '%d-%m-%y',
]

def _try_parse_date(value, formats, preferred_order=None):
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if not s or s.lower() in ['nan', 'none', 'nat', '-', '--', 'null', '']:
        return pd.NaT
    s = s.replace('：', ':').replace('　', ' ').replace('  ', ' ')

    ordered_formats = formats
    if preferred_order:
        year_first = []
        month_first = []
        day_first = []
        other = []
        for fmt in formats:
            if fmt.startswith('%Y') or fmt.startswith('%y'):
                year_first.append(fmt)
            elif fmt.startswith('%m') or fmt.startswith('%M'):
                if '%d' in fmt:
                    month_first.append(fmt)
                else:
                    other.append(fmt)
            elif fmt.startswith('%d'):
                day_first.append(fmt)
            else:
                other.append(fmt)

        if preferred_order == 'year_first':
            ordered_formats = year_first + month_first + day_first + other
        elif preferred_order == 'month_first':
            ordered_formats = month_first + year_first + day_first + other
        elif preferred_order == 'day_first':
            ordered_formats = day_first + year_first + month_first + other

    for fmt in ordered_formats:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 2000 or dt.year > 2100:
                continue
            return pd.Timestamp(dt)
        except (ValueError, TypeError):
            continue

    try:
        parts = s.replace('/', '-').replace('.', '-').split('-')
        if len(parts) == 3:
            y, m, d = None, None, None
            if len(parts[0]) == 4:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            elif len(parts[2]) == 4:
                if preferred_order == 'day_first':
                    y, m, d = int(parts[2]), int(parts[1]), int(parts[0])
                else:
                    y, m, d = int(parts[2]), int(parts[0]), int(parts[1])
                    if m > 12 or d > 31:
                        y, m, d = int(parts[2]), int(parts[1]), int(parts[0])
            if y and 1 <= m <= 12 and 1 <= d <= 31 and 2000 <= y <= 2100:
                try:
                    return pd.Timestamp(year=y, month=m, day=d)
                except (ValueError, TypeError):
                    pass
    except (ValueError, TypeError, IndexError):
        pass

    try:
        dayfirst_param = preferred_order == 'day_first'
        dt = pd.to_datetime(s, errors='raise', dayfirst=dayfirst_param)
        if 2000 <= dt.year <= 2100:
            return pd.Timestamp(dt)
    except (ValueError, TypeError):
        pass

    try:
        dt = pd.to_datetime(s, errors='raise', dayfirst=not (preferred_order == 'day_first'))
        if 2000 <= dt.year <= 2100:
            return pd.Timestamp(dt)
    except (ValueError, TypeError):
        pass

    try:
        num = int(s)
        dt = pd.to_datetime(num, unit='s', errors='raise')
        if 2000 <= dt.year <= 2100:
            return pd.Timestamp(dt)
        dt = pd.to_datetime(num, unit='ms', errors='raise')
        if 2000 <= dt.year <= 2100:
            return pd.Timestamp(dt)
    except (ValueError, TypeError, OverflowError):
        pass

    return pd.NaT

def parse_date_smart(series):
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return pd.Series([pd.NaT] * len(series), index=series.index, dtype='datetime64[ns]')

    count_y_first = 0
    count_m_first = 0
    count_d_first = 0

    for val in non_null.head(20):
        s = val.strip()
        if len(s) >= 10 and (s[:4].isdigit() or s[:5].replace('-', '').replace('/', '').isdigit()):
            if s[:4].isdigit():
                try:
                    y = int(s[:4])
                    if 2000 <= y <= 2100:
                        count_y_first += 1
                        continue
                except ValueError:
                    pass
        if len(s) >= 8 and any(sep in s for sep in ['/', '-', '.']):
            parts = s.replace('/', '-').replace('.', '-').split('-')
            if len(parts) == 3:
                try:
                    p0, p1, p2 = int(parts[0]), int(parts[1]), int(parts[2])
                    if len(parts[2]) == 4 and 2000 <= p2 <= 2100:
                        is_possible_mmdd = (1 <= p0 <= 12) and (1 <= p1 <= 31)
                        is_possible_ddmm = (1 <= p1 <= 12) and (1 <= p0 <= 31)
                        is_definitely_mmdd = (p0 <= 12) and (p1 > 12)
                        is_definitely_ddmm = (p1 <= 12) and (p0 > 12)

                        if is_definitely_mmdd:
                            count_m_first += 2
                        elif is_definitely_ddmm:
                            count_d_first += 2
                        elif is_possible_mmdd and is_possible_ddmm:
                            pass
                        elif is_possible_mmdd:
                            count_m_first += 1
                        elif is_possible_ddmm:
                            count_d_first += 1
                except ValueError:
                    pass

    preferred_order = None
    if count_y_first >= max(count_m_first, count_d_first, 1):
        preferred_order = 'year_first'
    elif count_m_first > count_d_first:
        preferred_order = 'month_first'
    elif count_d_first > count_m_first:
        preferred_order = 'day_first'

    result = []
    for v in series:
        dt = _try_parse_date(v, DATE_FORMATS, preferred_order)
        result.append(dt)

    return pd.Series(result, index=series.index, dtype='datetime64[ns]')

def minute_to_hhmm(min_val):
    if pd.isna(min_val):
        return ''
    m = int(min_val)
    return f"{m//60:02d}:{m%60:02d}"

def load_dataframe(file):
    content = file.read()
    encoding = detect_encoding(content)
    content_io = io.BytesIO(content)
    df = pd.read_csv(content_io, encoding=encoding)
    return df

def auto_map_fields(uploaded_cols):
    mapping = {}
    lower_cols = {col.lower(): col for col in uploaded_cols}
    for std_field, synonyms in FIELD_SYNONYMS.items():
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower in lower_cols:
                mapping[std_field] = lower_cols[syn_lower]
                break
            for orig_col in uploaded_cols:
                if syn_lower in orig_col.lower():
                    mapping[std_field] = orig_col
                    break
            if std_field in mapping:
                break
    return mapping

@st.cache_data(show_spinner=False)
def process_data(raw_df, field_mapping):
    df = pd.DataFrame()
    inv_map = {v: k for k, v in field_mapping.items()}

    for orig_col in raw_df.columns:
        std_field = inv_map.get(orig_col)
        if std_field:
            df[std_field] = raw_df[orig_col].copy()
        else:
            df[f"ext_{orig_col}"] = raw_df[orig_col].copy()

    for std_field in REQUIRED_FIELDS:
        if std_field not in df.columns:
            df[std_field] = np.nan

    anomalies = []

    df['record_date_parsed'] = parse_date_smart(df['record_date'])
    bad_date = df['record_date_parsed'].isna()
    for idx in df[bad_date].index:
        anomalies.append({
            '行号': idx + 2,
            '类型': '日期解析失败',
            '原始值': str(df.loc[idx, 'record_date']),
            '说明': '无法识别的日期格式，建议使用 YYYY-MM-DD 格式'
        })

    df['order_min_parsed'] = df['order_minute'].apply(parse_time_to_minute)
    df['finish_min_parsed'] = df['finish_minute'].apply(parse_time_to_minute)
    bad_order = df['order_min_parsed'].isna() & df['order_minute'].notna()
    bad_finish = df['finish_min_parsed'].isna() & df['finish_minute'].notna()
    for idx in df[bad_order].index:
        anomalies.append({
            '行号': idx + 2,
            '类型': '下单时间解析失败',
            '原始值': str(df.loc[idx, 'order_minute']),
            '说明': '建议使用 HH:MM 或分钟数(0-1440)格式'
        })
    for idx in df[bad_finish].index:
        anomalies.append({
            '行号': idx + 2,
            '类型': '完成时间解析失败',
            '原始值': str(df.loc[idx, 'finish_minute']),
            '说明': '建议使用 HH:MM 或分钟数(0-1440)格式'
        })

    df['brew_duration'] = np.nan
    valid_time = df['order_min_parsed'].notna() & df['finish_min_parsed'].notna()
    df.loc[valid_time, 'brew_duration'] = (
        df.loc[valid_time, 'finish_min_parsed'] - df.loc[valid_time, 'order_min_parsed']
    )
    neg_duration = df['brew_duration'] < 0
    df.loc[neg_duration & valid_time, 'brew_duration'] += 1440

    df['remake_flag_parsed'] = df['remake_flag'].apply(parse_remake_flag)

    bad_brew = df['brew_duration'] > 120
    for idx in df[bad_brew].index:
        anomalies.append({
            '行号': idx + 2,
            '类型': '出杯时长异常',
            '原始值': f"{df.loc[idx, 'brew_duration']:.0f}分钟",
            '说明': '出杯时长超过2小时，建议核查'
        })
    zero_brew = (df['brew_duration'] == 0) & valid_time
    for idx in df[zero_brew].index:
        anomalies.append({
            '行号': idx + 2,
            '类型': '出杯时长为0',
            '原始值': '0分钟',
            '说明': '下单与完成时间相同，可能为数据录入问题'
        })

    df['is_remake'] = df['remake_flag_parsed'] == 1

    anomaly_df = pd.DataFrame(anomalies) if anomalies else pd.DataFrame(columns=['行号', '类型', '原始值', '说明'])

    return df, anomaly_df

def filter_dataframe(df, date_range, stores, drinks, baristas, remake_only):
    mask = pd.Series(True, index=df.index)
    if date_range and len(date_range) == 2:
        mask &= (df['record_date_parsed'] >= pd.Timestamp(date_range[0])) & \
                (df['record_date_parsed'] <= pd.Timestamp(date_range[1]) + timedelta(days=1) - timedelta(seconds=1))
    if stores:
        mask &= df['store_name'].astype(str).isin(stores)
    if drinks:
        mask &= df['drink_name'].astype(str).isin(drinks)
    if baristas:
        mask &= df['barista_name'].astype(str).isin(baristas)
    if remake_only:
        mask &= df['is_remake']
    return df[mask].copy()

def generate_html_report(df_filtered, df_raw, anomaly_df, summary_metrics, prep_suggestions):
    metrics_html = ""
    for k, v in summary_metrics.items():
        metrics_html += f'<div class="metric-card"><div class="metric-label">{k}</div><div class="metric-value">{v}</div></div>'

    store_table = ""
    drink_table = ""
    barista_table = ""
    anomaly_table = ""
    suggestion_section = ""
    daily_chart_data = ""
    remake_chart_data = ""

    valid_df = df_filtered.dropna(subset=['brew_duration', 'record_date_parsed'])

    if len(valid_df) > 0:
        store_stats = valid_df.groupby('store_name').agg(
            订单数=('brew_duration', 'count'),
            平均出杯时长=('brew_duration', 'mean'),
            重做率=('is_remake', 'mean')
        ).reset_index()
        store_stats.columns = ['门店', '订单数', '平均时长(分)', '重做率(%)']
        store_stats['平均时长(分)'] = store_stats['平均时长(分)'].apply(lambda x: f"{x:.1f}")
        store_stats['重做率(%)'] = store_stats['重做率(%)'].apply(lambda x: f"{x*100:.1f}")
        store_stats = store_stats.sort_values('订单数', ascending=False)
        store_table = store_stats.to_html(index=False, classes='data-table')

        drink_stats = df_filtered.groupby('drink_name').agg(
            订单数=('is_remake', 'count'),
            重做数=('is_remake', 'sum'),
            重做率=('is_remake', 'mean')
        ).reset_index()
        drink_stats.columns = ['饮品', '订单数', '重做数', '重做率(%)']
        drink_stats['重做率(%)'] = drink_stats['重做率(%)'].apply(lambda x: f"{x*100:.1f}")
        drink_stats = drink_stats.sort_values('重做率(%)', ascending=False, key=lambda x: x.astype(float))
        drink_table = drink_stats.to_html(index=False, classes='data-table')

        barista_stats = valid_df.groupby('barista_name').agg(
            出杯数=('brew_duration', 'count'),
            平均出杯时长=('brew_duration', 'mean'),
            重做率=('is_remake', 'mean')
        ).reset_index()
        barista_stats.columns = ['咖啡师', '出杯数', '平均时长(分)', '重做率(%)']
        barista_stats['平均时长(分)'] = barista_stats['平均时长(分)'].apply(lambda x: f"{x:.1f}")
        barista_stats['重做率(%)'] = barista_stats['重做率(%)'].apply(lambda x: f"{x*100:.1f}")
        barista_stats = barista_stats.sort_values('出杯数', ascending=False)
        barista_table = barista_stats.to_html(index=False, classes='data-table')

        valid_dated = valid_df.copy()
        valid_dated['date_only'] = valid_dated['record_date_parsed'].dt.date
        daily_stats = valid_dated.groupby('date_only').agg(
            平均出杯时长=('brew_duration', 'mean'),
            订单量=('brew_duration', 'count'),
            重做率=('is_remake', 'mean')
        ).reset_index()
        daily_stats = daily_stats.tail(14)
        if len(daily_stats) > 0:
            daily_labels = [str(d) for d in daily_stats['date_only'].tolist()]
            daily_durations = [f"{x:.1f}" for x in daily_stats['平均出杯时长'].tolist()]
            daily_volumes = daily_stats['订单量'].tolist()
            daily_remakes = [f"{x*100:.1f}" for x in daily_stats['重做率'].tolist()]
            daily_chart_data = f"""
            <div class="chart-container">
                <h3>📈 近14天效率趋势</h3>
                <div class="chart-row">
                    <div class="mini-chart">
                        <div class="chart-title">平均出杯时长（分）</div>
                        <div class="bar-chart">
                            {''.join(f'<div class="bar-item" style="height:{min(100, float(v)*3)}%"><span class="bar-label">{v}</span></div>' for v in daily_durations)}
                        </div>
                        <div class="chart-labels">{' '.join(f'<span class="lbl">{l[5:]}</span>' for l in daily_labels)}</div>
                    </div>
                    <div class="mini-chart">
                        <div class="chart-title">订单量</div>
                        <div class="bar-chart">
                            {''.join(f'<div class="bar-item green" style="height:{min(100, v*2)}%"><span class="bar-label">{v}</span></div>' for v in daily_volumes)}
                        </div>
                        <div class="chart-labels">{' '.join(f'<span class="lbl">{l[5:]}</span>' for l in daily_labels)}</div>
                    </div>
                    <div class="mini-chart">
                        <div class="chart-title">重做率（%）</div>
                        <div class="bar-chart">
                            {''.join(f'<div class="bar-item red" style="height:{min(100, float(v)*5)}%"><span class="bar-label">{v}</span></div>' for v in daily_remakes)}
                        </div>
                        <div class="chart-labels">{' '.join(f'<span class="lbl">{l[5:]}</span>' for l in daily_labels)}</div>
                    </div>
                </div>
            </div>
            """

    if len(df_filtered) > 0:
        drink_remake = df_filtered.groupby('drink_name').agg(
            订单数=('is_remake', 'count'),
            重做率=('is_remake', 'mean')
        ).reset_index()
        drink_remake = drink_remake.sort_values('重做率', ascending=False).head(10)
        if len(drink_remake) > 0:
            remake_labels = drink_remake['drink_name'].tolist()
            remake_values = [f"{x*100:.1f}" for x in drink_remake['重做率'].tolist()]
            remake_orders = drink_remake['订单数'].tolist()
            remake_chart_data = f"""
            <div class="chart-container">
                <h3>🔄 饮品重做率 Top 10</h3>
                <div class="remake-list">
                    {''.join(f'''
                    <div class="remake-row">
                        <span class="remake-name">{i+1}. {lbl}</span>
                        <span class="remake-count">({ord}单)</span>
                        <div class="remake-bar-bg">
                            <div class="remake-bar {('warn' if float(val) > 15 else '')}" style="width:{min(100, float(val)*2)}%"></div>
                        </div>
                        <span class="remake-pct">{val}%</span>
                    </div>
                    ''' for i, (lbl, val, ord) in enumerate(zip(remake_labels, remake_values, remake_orders)))}
                </div>
            </div>
            """

    if len(anomaly_df) > 0:
        anomaly_table = anomaly_df.to_html(index=False, classes='data-table anomaly-table')

    if prep_suggestions:
        suggestion_items = ""
        for i, s in enumerate(prep_suggestions, 1):
            suggestion_items += f"""
            <div class="suggestion-card">
                <div class="suggestion-title">建议 {i}：优化 [{s['饮品']}] 备料方案</div>
                <div class="suggestion-metrics">
                    <span class="chip">📦 订单量: {s['订单量']}</span>
                    <span class="chip">⏱️ 时长: {s['平均出杯时长(分钟)']}分钟</span>
                    <span class="chip">🔄 重做率: {s['重做率']}</span>
                </div>
                <div class="suggestion-content">
                    <strong>问题点：</strong>{s['问题点']}<br>
                    <strong>建议措施：</strong>{s['建议']}
                </div>
            </div>
            """
        suggestion_section = f"""
        <h2>💡 备料调整建议</h2>
        <div class="suggestion-grid">{suggestion_items}</div>
        """

    detail_table = ""
    if len(df_filtered) > 0:
        display_cols = [c for c in ['record_date', 'store_name', 'drink_name', 'barista_name',
                                   'order_minute', 'finish_minute', 'brew_duration', 'remake_flag', 'note']
                        if c in df_filtered.columns]
        detail_df = df_filtered[display_cols].copy()
        if 'brew_duration' in detail_df.columns:
            detail_df['brew_duration'] = detail_df['brew_duration'].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else ""
            )
        detail_table = detail_df.head(100).to_html(index=False, classes='data-table detail-table')
        detail_count = len(df_filtered)
        detail_note = f"<p style='color:#666;font-size:12px;margin-top:5px;'>* 显示前 100 条，共 {detail_count} 条记录</p>"

    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <title>咖啡门店出杯效率分析报告</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
            h1 {{ color: #8B4513; border-bottom: 3px solid #D2691E; padding-bottom: 10px; }}
            h2 {{ color: #A0522D; margin-top: 35px; border-left: 4px solid #D2691E; padding-left: 12px; }}
            h3 {{ color: #8B4513; margin-top: 20px; }}
            .metrics-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
            .metric-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border-left: 4px solid #D2691E; }}
            .metric-label {{ color: #666; font-size: 14px; margin-bottom: 8px; }}
            .metric-value {{ color: #8B4513; font-size: 28px; font-weight: bold; }}
            .data-table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }}
            .data-table th {{ background: #8B4513; color: white; padding: 12px; text-align: left; font-weight: 600; }}
            .data-table td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
            .data-table tr:hover {{ background: #FFF8DC; }}
            .anomaly-table th {{ background: #B22222; }}
            .anomaly-table tr:hover {{ background: #FFE4E1; }}
            .detail-table {{ font-size: 12px; }}
            .detail-table th {{ padding: 8px 10px; }}
            .detail-table td {{ padding: 6px 10px; }}
            .chart-container {{ background: white; padding: 20px; border-radius: 10px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
            .chart-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
            .mini-chart {{ flex: 1; min-width: 280px; }}
            .chart-title {{ font-size: 13px; color: #666; margin-bottom: 8px; text-align: center; }}
            .bar-chart {{ display: flex; align-items: flex-end; gap: 4px; height: 150px; padding: 10px 5px 0; border-bottom: 1px solid #eee; }}
            .bar-item {{ flex: 1; background: linear-gradient(to top, #D2691E, #DEB887); border-radius: 4px 4px 0 0; position: relative; min-height: 4px; }}
            .bar-item.green {{ background: linear-gradient(to top, #228B22, #90EE90); }}
            .bar-item.red {{ background: linear-gradient(to top, #DC143C, #FFB6C1); }}
            .bar-label {{ position: absolute; top: -18px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #666; white-space: nowrap; }}
            .chart-labels {{ display: flex; justify-content: space-between; margin-top: 5px; }}
            .lbl {{ font-size: 10px; color: #999; }}
            .remake-list {{ display: flex; flex-direction: column; gap: 8px; margin-top: 10px; }}
            .remake-row {{ display: flex; align-items: center; gap: 12px; }}
            .remake-name {{ width: 140px; font-weight: 500; }}
            .remake-count {{ color: #888; font-size: 12px; width: 50px; }}
            .remake-bar-bg {{ flex: 1; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }}
            .remake-bar {{ height: 100%; background: linear-gradient(to right, #FFA500, #FF6347); transition: width 0.3s; }}
            .remake-bar.warn {{ background: linear-gradient(to right, #DC143C, #8B0000); }}
            .remake-pct {{ width: 60px; text-align: right; font-weight: 600; }}
            .suggestion-grid {{ display: grid; gap: 15px; margin: 15px 0; }}
            .suggestion-card {{ background: white; padding: 18px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); border-left: 4px solid #D2691E; }}
            .suggestion-title {{ font-size: 16px; font-weight: 600; color: #8B4513; margin-bottom: 10px; }}
            .suggestion-metrics {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 10px; }}
            .chip {{ background: #FFF8DC; padding: 4px 10px; border-radius: 12px; font-size: 12px; color: #8B4513; }}
            .suggestion-content {{ color: #555; line-height: 1.7; }}
            .section-note {{ background: #E8F4FD; border-left: 4px solid #2196F3; padding: 12px 15px; margin: 15px 0; border-radius: 4px; color: #1565C0; }}
            .report-footer {{ margin-top: 40px; color: #999; font-size: 12px; text-align: center; border-top: 1px solid #ddd; padding-top: 20px; }}
            @media print {{
                body {{ margin: 15px; }}
                .data-table {{ page-break-inside: avoid; }}
                .suggestion-card {{ page-break-inside: avoid; }}
            }}
        </style>
    </head>
    <body>
        <h1>☕ 咖啡门店出杯效率分析报告</h1>
        <p><strong>生成时间：</strong>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 核心指标</h2>
        <div class="metrics-container">{metrics_html}</div>

        <h2>📋 数据概览</h2>
        <table class="data-table">
            <tr><th style="width:40%">项目</th><th>数值</th></tr>
            <tr><td>总记录数</td><td>{len(df_raw)}</td></tr>
            <tr><td>筛选后记录数</td><td>{len(df_filtered)}</td></tr>
            <tr><td>有效时长记录数</td><td>{len(valid_df)}</td></tr>
            <tr><td>异常记录数</td><td>{len(anomaly_df)}</td></tr>
            <tr><td>涉及门店数</td><td>{df_filtered['store_name'].nunique() if len(df_filtered) > 0 else 0}</td></tr>
            <tr><td>涉及饮品数</td><td>{df_filtered['drink_name'].nunique() if len(df_filtered) > 0 else 0}</td></tr>
            <tr><td>涉及咖啡师数</td><td>{df_filtered['barista_name'].nunique() if len(df_filtered) > 0 else 0}</td></tr>
        </table>

        {daily_chart_data}

        {remake_chart_data}

        <h2>🏪 门店统计</h2>
        {store_table if store_table else '<div class="section-note">暂无有效门店数据</div>'}

        <h2>🥤 饮品统计</h2>
        {drink_table if drink_table else '<div class="section-note">暂无有效饮品数据</div>'}

        <h2>👨‍🍳 咖啡师统计</h2>
        {barista_table if barista_table else '<div class="section-note">暂无有效咖啡师数据</div>'}

        {suggestion_section}

        <h2>⚠️ 异常明细</h2>
        {anomaly_table if anomaly_table else '<div class="section-note">✅ 未发现异常数据</div>'}

        <h2>📋 明细数据</h2>
        {detail_table if detail_table else '<div class="section-note">暂无明细数据</div>'}
        {detail_note if detail_table else ''}

        <div class="report-footer">
            本报告由咖啡门店出杯效率分析看板自动生成 | 数据仅保存在会话缓存中
        </div>
    </body>
    </html>
    """
    return html

def get_prep_suggestions(df):
    suggestions = []
    valid = df.dropna(subset=['brew_duration', 'record_date_parsed', 'order_min_parsed'])

    if len(valid) == 0:
        return suggestions

    valid['hour'] = (valid['order_min_parsed'] // 60).astype(int)
    valid['is_peak'] = valid['hour'].apply(lambda x: (7 <= x <= 10) or (12 <= x <= 14) or (17 <= x <= 19))

    overall_avg = valid['brew_duration'].mean()

    drink_stats = valid.groupby('drink_name').agg(
        订单数=('drink_name', 'count'),
        平均出杯时长=('brew_duration', 'mean'),
        重做率=('is_remake', 'mean'),
        高峰占比=('is_peak', 'mean')
    ).reset_index()

    hot_drinks = drink_stats[drink_stats['订单数'] >= drink_stats['订单数'].quantile(0.6)]

    for _, row in hot_drinks.iterrows():
        drink = row['drink_name']
        reasons = []

        if row['平均出杯时长'] > overall_avg * 1.3:
            reasons.append(f"出杯慢（平均{row['平均出杯时长']:.1f}分钟，高于均值30%+）")
        if row['重做率'] > 0.15:
            reasons.append(f"重做率高（{row['重做率']*100:.1f}%）")
        if row['高峰占比'] > 0.6:
            reasons.append(f"高峰订单集中（{row['高峰占比']*100:.0f}%订单在高峰时段）")

        if reasons:
            drink_peak_hours = valid[valid['drink_name'] == drink].groupby('hour')['drink_name'].count()
            if len(drink_peak_hours) > 0:
                top_hour = drink_peak_hours.idxmax()
                peak_hint = f"建议{top_hour-1:02d}:00前备好{drink}半成品"
            else:
                peak_hint = ""

            suggestions.append({
                '饮品': drink,
                '订单量': int(row['订单数']),
                '平均出杯时长(分钟)': f"{row['平均出杯时长']:.1f}",
                '重做率': f"{row['重做率']*100:.1f}%",
                '问题点': '；'.join(reasons),
                '建议': peak_hint or "建议提前准备半成品，优化制作流程"
            })

    return suggestions

def main():
    st.title("☕ 咖啡门店出杯效率分析看板")
    st.markdown("---")

    if 'raw_df' not in st.session_state:
        st.session_state.raw_df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'anomaly_df' not in st.session_state:
        st.session_state.anomaly_df = None
    if 'field_mapping' not in st.session_state:
        st.session_state.field_mapping = None

    with st.sidebar:
        st.header("📁 数据上传")
        uploaded_file = st.file_uploader(
            "上传 CSV 文件（订单流水/出杯记录/顾客反馈）",
            type=['csv'],
            help="支持字段：record_date, store_name, drink_name, order_minute, finish_minute, remake_flag, barista_name, note"
        )

        if uploaded_file is not None:
            try:
                raw_df = load_dataframe(uploaded_file)
                st.session_state.raw_df = raw_df
                st.success(f"✅ 文件加载成功，共 {len(raw_df)} 行 / {len(raw_df.columns)} 列")

                with st.expander("🔧 字段映射配置", expanded=True):
                    st.markdown("**系统已自动匹配字段，可手动调整：**")
                    auto_map = auto_map_fields(raw_df.columns.tolist())
                    mapping = {}
                    for std_field, label in REQUIRED_FIELDS.items():
                        options = ['（无）'] + raw_df.columns.tolist()
                        default_idx = 0
                        if std_field in auto_map:
                            if auto_map[std_field] in raw_df.columns:
                                default_idx = raw_df.columns.tolist().index(auto_map[std_field]) + 1
                        selected = st.selectbox(
                            f"{label} (`{std_field}`)",
                            options,
                            index=default_idx,
                            key=f"map_{std_field}"
                        )
                        if selected != '（无）':
                            mapping[std_field] = selected

                    if st.button("✅ 确认字段映射并处理数据", type="primary", use_container_width=True):
                        if not mapping:
                            st.error("请至少映射一个字段")
                        else:
                            with st.spinner("正在处理数据..."):
                                processed_df, anomaly_df = process_data(raw_df, mapping)
                                st.session_state.processed_df = processed_df
                                st.session_state.anomaly_df = anomaly_df
                                st.session_state.field_mapping = mapping
                                st.success(f"✅ 数据处理完成，识别到 {len(anomaly_df)} 条异常")

            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")

        st.markdown("---")
        st.header("📖 使用说明")
        with st.expander("查看详细说明"):
            st.markdown("""
            **支持的字段：**
            - `record_date`：订单日期（支持YYYY-MM-DD、YYYY/MM/DD等多种格式）
            - `store_name`：门店名称
            - `drink_name`：饮品名称
            - `order_minute`：下单时间（支持HH:MM或分钟数0-1440）
            - `finish_minute`：完成时间（同上）
            - `remake_flag`：重做标记（1/0、是/否、True/False）
            - `barista_name`：咖啡师姓名
            - `note`：备注

            **日期解析提示：**
            - 推荐格式：YYYY-MM-DD
            - 其他常见格式也可自动识别
            - 无法解析的日期会在异常明细中列出
            """)

    if st.session_state.processed_df is None:
        st.info("👈 请先在左侧上传 CSV 数据文件并完成字段映射")
        st.markdown("### 📋 示例数据格式")
        sample_data = {
            'record_date': ['2026-06-01', '2026-06-01', '2026-06-02', '2026-06-02', '2026-06-03'],
            'store_name': ['中心店', '中心店', '科技园店', '中心店', '科技园店'],
            'drink_name': ['拿铁', '美式', '拿铁', '卡布奇诺', '美式'],
            'order_minute': ['08:15', '08:30', '12:05', '09:00', '18:20'],
            'finish_minute': ['08:20', '08:33', '12:12', '09:06', '18:24'],
            'remake_flag': [0, 1, 0, 0, 0],
            'barista_name': ['张三', '李四', '王五', '张三', '王五'],
            'note': ['', '奶泡不合格重做', '', '', '']
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        return

    df = st.session_state.processed_df
    anomaly_df = st.session_state.anomaly_df
    mapping = st.session_state.field_mapping

    st.header("🎯 数据概览 & 筛选")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    col_info1.metric("📝 总记录数", len(df))
    col_info2.metric("🏪 门店数", df['store_name'].nunique())
    col_info3.metric("🥤 饮品数", df['drink_name'].nunique())
    col_info4.metric("👨‍🍳 咖啡师数", df['barista_name'].nunique())

    st.markdown("---")

    with st.expander("🔍 筛选条件", expanded=True):
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            valid_dates = df['record_date_parsed'].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                date_range = st.date_input(
                    "📅 日期范围",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
                st.warning("无有效日期数据")

            store_options = sorted(df['store_name'].dropna().unique().tolist())
            selected_stores = st.multiselect("🏪 门店", store_options, default=store_options)

            drink_options = sorted(df['drink_name'].dropna().unique().tolist())
            selected_drinks = st.multiselect("🥤 饮品", drink_options, default=[])

        with fcol2:
            barista_options = sorted(df['barista_name'].dropna().unique().tolist())
            selected_baristas = st.multiselect("👨‍🍳 咖啡师", barista_options, default=[])
            remake_only = st.checkbox("🔄 仅显示重做记录", value=False)

    df_filtered = filter_dataframe(df, date_range, selected_stores, selected_drinks, selected_baristas, remake_only)

    st.markdown(f"**筛选结果：** {len(df_filtered)} 条记录")

    valid_df = df_filtered.dropna(subset=['brew_duration'])

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 核心分析", "🔄 重做趋势", "👨‍🍳 咖啡师负载", "⚠️ 异常明细", "💡 备料建议"
    ])

    with tab1:
        st.subheader("📊 门店出杯时长分布")

        if len(valid_df) > 0:
            fig_duration_box = px.box(
                valid_df,
                x='store_name',
                y='brew_duration',
                color='store_name',
                title='各门店出杯时长分布（箱线图）',
                labels={'store_name': '门店', 'brew_duration': '出杯时长（分钟）'},
                points='outliers'
            )
            fig_duration_box.update_layout(height=450, showlegend=False)
            st.plotly_chart(fig_duration_box, use_container_width=True)

            col_hist1, col_hist2 = st.columns(2)
            with col_hist1:
                fig_hist = px.histogram(
                    valid_df,
                    x='brew_duration',
                    nbins=50,
                    title='整体出杯时长分布',
                    labels={'brew_duration': '出杯时长（分钟）', 'count': '订单数'}
                )
                fig_hist.update_layout(height=350)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_hist2:
                store_avg = valid_df.groupby('store_name')['brew_duration'].agg(['mean', 'median', 'count']).reset_index()
                store_avg.columns = ['门店', '平均时长', '中位数时长', '订单量']
                store_avg = store_avg.sort_values('平均时长', ascending=False)
                fig_store = px.bar(
                    store_avg,
                    x='门店',
                    y='平均时长',
                    text='平均时长',
                    title='各门店平均出杯时长',
                    color='平均时长',
                    color_continuous_scale='Reds'
                )
                fig_store.update_traces(texttemplate='%{text:.1f}分', textposition='outside')
                fig_store.update_layout(height=350)
                st.plotly_chart(fig_store, use_container_width=True)

            st.markdown("---")
            st.subheader("📈 近14天出杯效率变化")
            valid_dated = valid_df.dropna(subset=['record_date_parsed'])
            if len(valid_dated) > 0:
                valid_dated['date_only'] = valid_dated['record_date_parsed'].dt.date
                daily_stats = valid_dated.groupby('date_only').agg(
                    平均出杯时长=('brew_duration', 'mean'),
                    订单量=('brew_duration', 'count'),
                    重做率=('is_remake', 'mean')
                ).reset_index()
                daily_stats = daily_stats.sort_values('date_only').tail(14)

                fig_daily = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=('平均出杯时长趋势', '日订单量 & 重做率趋势')
                )
                fig_daily.add_trace(
                    go.Scatter(x=daily_stats['date_only'], y=daily_stats['平均出杯时长'],
                               mode='lines+markers', name='平均出杯时长(分)',
                               line=dict(color='#D2691E', width=2)),
                    row=1, col=1
                )
                fig_daily.add_trace(
                    go.Bar(x=daily_stats['date_only'], y=daily_stats['订单量'],
                           name='订单量', marker_color='#8B4513', opacity=0.7),
                    row=2, col=1
                )
                fig_daily.add_trace(
                    go.Scatter(x=daily_stats['date_only'], y=daily_stats['重做率']*100,
                               mode='lines+markers', name='重做率(%)',
                               line=dict(color='#FF4500', width=2), yaxis='y2'),
                    row=2, col=1
                )
                fig_daily.update_layout(height=550, showlegend=True)
                st.plotly_chart(fig_daily, use_container_width=True)
        else:
            st.warning("无有效出杯时长数据")

    with tab2:
        st.subheader("🔄 饮品重做率趋势")

        valid_dated = df_filtered.dropna(subset=['record_date_parsed'])
        if len(valid_dated) > 0:
            valid_dated['date_only'] = valid_dated['record_date_parsed'].dt.date

            overall_remake_rate = df_filtered['is_remake'].mean() * 100 if len(df_filtered) > 0 else 0
            st.metric("整体重做率", f"{overall_remake_rate:.2f}%", delta=None)

            drink_remake = df_filtered.groupby('drink_name').agg(
                订单数=('is_remake', 'count'),
                重做数=('is_remake', 'sum'),
                重做率=('is_remake', 'mean')
            ).reset_index()
            drink_remake = drink_remake.sort_values('重做率', ascending=False)
            drink_remake['重做率%'] = drink_remake['重做率'] * 100
            drink_remake['样本量'] = drink_remake['订单数'].apply(lambda x: '⚠️ 小样本' if x < 3 else '✅ 可信')

            if len(drink_remake) > 0:
                fig_remake = px.bar(
                    drink_remake,
                    x='drink_name',
                    y='重做率%',
                    text='重做率%',
                    hover_data=['订单数', '重做数', '样本量'],
                    title='各饮品重做率排行（含全部饮品）',
                    color='样本量',
                    color_discrete_map={'✅ 可信': '#D2691E', '⚠️ 小样本': '#FFA07A'},
                    pattern_shape='样本量',
                    pattern_shape_map={'✅ 可信': '', '⚠️ 小样本': '/'}
                )
                fig_remake.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig_remake.update_layout(height=450, legend_title='数据可靠性')
                st.plotly_chart(fig_remake, use_container_width=True)

                st.markdown("### 📋 饮品重做率详细表")
                display_remake = drink_remake.copy()
                display_remake['重做率%'] = display_remake['重做率%'].apply(lambda x: f"{x:.1f}%")
                display_remake = display_remake[['drink_name', '订单数', '重做数', '重做率%', '样本量']]
                display_remake.columns = ['饮品名称', '总订单数', '重做次数', '重做率', '数据可靠性']
                st.dataframe(
                    display_remake,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "数据可靠性": st.column_config.Column(
                            "数据可靠性",
                            width="small"
                        )
                    }
                )
                st.caption("⚠️ 小样本标记表示订单数少于3单，重做率可能不具备统计代表性")
            else:
                st.info("暂无饮品重做数据")

            daily_remake = valid_dated.groupby('date_only').agg(
                订单数=('is_remake', 'count'),
                重做数=('is_remake', 'sum'),
                重做率=('is_remake', 'mean')
            ).reset_index()
            daily_remake = daily_remake.sort_values('date_only')
            daily_remake['重做率%'] = daily_remake['重做率'] * 100

            fig_daily_remake = go.Figure()
            fig_daily_remake.add_trace(
                go.Bar(x=daily_remake['date_only'], y=daily_remake['订单数'],
                       name='总订单', marker_color='#D2B48C', opacity=0.6)
            )
            fig_daily_remake.add_trace(
                go.Bar(x=daily_remake['date_only'], y=daily_remake['重做数'],
                       name='重做数', marker_color='#FF6347')
            )
            fig_daily_remake.add_trace(
                go.Scatter(x=daily_remake['date_only'], y=daily_remake['重做率%'],
                           mode='lines+markers', name='重做率(%)',
                           line=dict(color='#8B0000', width=3),
                           yaxis='y2')
            )
            fig_daily_remake.update_layout(
                barmode='overlay',
                title='每日订单量与重做率趋势',
                height=450,
                yaxis=dict(title='订单数'),
                yaxis2=dict(title='重做率(%)', overlaying='y', side='right', range=[0, max(50, daily_remake['重做率%'].max()*1.2 if len(daily_remake)>0 else 50)])
            )
            st.plotly_chart(fig_daily_remake, use_container_width=True)

    with tab3:
        st.subheader("👨‍🍳 咖啡师负载与绩效分析")

        if len(valid_df) > 0:
            barista_stats = valid_df.groupby('barista_name').agg(
                出杯数=('brew_duration', 'count'),
                平均出杯时长=('brew_duration', 'mean'),
                总出杯时间=('brew_duration', 'sum'),
                重做率=('is_remake', 'mean')
            ).reset_index()
            barista_stats = barista_stats.sort_values('出杯数', ascending=False)
            barista_stats['重做率%'] = barista_stats['重做率'] * 100

            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fig_barista_count = px.bar(
                    barista_stats,
                    x='barista_name',
                    y='出杯数',
                    text='出杯数',
                    title='咖啡师出杯量排行',
                    color='出杯数',
                    color_continuous_scale='Blues'
                )
                fig_barista_count.update_layout(height=400)
                st.plotly_chart(fig_barista_count, use_container_width=True)

            with col_b2:
                fig_barista_matrix = px.scatter(
                    barista_stats,
                    x='平均出杯时长',
                    y='重做率%',
                    size='出杯数',
                    color='出杯数',
                    hover_name='barista_name',
                    hover_data=['出杯数', '平均出杯时长', '重做率%'],
                    title='咖啡师效率矩阵（速度 vs 质量）',
                    color_continuous_scale='Viridis',
                    size_max=50
                )
                fig_barista_matrix.update_layout(height=400)
                fig_barista_matrix.add_hline(
                    y=barista_stats['重做率%'].mean(),
                    line_dash="dash", line_color="red",
                    annotation_text=f"平均重做率: {barista_stats['重做率%'].mean():.1f}%"
                )
                fig_barista_matrix.add_vline(
                    x=barista_stats['平均出杯时长'].mean(),
                    line_dash="dash", line_color="red",
                    annotation_text=f"平均时长: {barista_stats['平均出杯时长'].mean():.1f}分"
                )
                st.plotly_chart(fig_barista_matrix, use_container_width=True)

            st.markdown("---")
            st.markdown("### 📋 咖啡师详细统计")
            display_stats = barista_stats.copy()
            display_stats['平均出杯时长'] = display_stats['平均出杯时长'].apply(lambda x: f"{x:.1f}分钟")
            display_stats['总出杯时间'] = display_stats['总出杯时间'].apply(lambda x: f"{x:.0f}分钟")
            display_stats['重做率'] = display_stats['重做率%'].apply(lambda x: f"{x:.1f}%")
            display_stats = display_stats.drop(columns=['重做率%'])
            display_stats.columns = ['咖啡师', '出杯数', '平均出杯时长', '总出杯时间', '重做率']
            st.dataframe(display_stats, use_container_width=True, hide_index=True)

            st.markdown("---")
            valid_hour = valid_df.copy()
            valid_hour['hour'] = (valid_hour['order_min_parsed'] // 60).astype(int)
            hour_load = valid_hour.groupby(['hour', 'barista_name']).size().reset_index(name='出杯数')

            fig_hourmap = px.density_heatmap(
                hour_load,
                x='hour',
                y='barista_name',
                z='出杯数',
                title='各时段咖啡师出杯热力图',
                labels={'hour': '小时', 'barista_name': '咖啡师', '出杯数': '出杯数'},
                color_continuous_scale='YlOrRd',
                nbinsx=24
            )
            fig_hourmap.update_layout(height=450)
            st.plotly_chart(fig_hourmap, use_container_width=True)

    with tab4:
        st.subheader("⚠️ 异常行明细")

        total_anomalies = len(anomaly_df)
        st.metric("异常记录总数", total_anomalies)

        if total_anomalies > 0:
            col_a1, col_a2 = st.columns([1, 2])
            with col_a1:
                type_counts = anomaly_df['类型'].value_counts().reset_index()
                type_counts.columns = ['异常类型', '数量']
                fig_anom = px.pie(
                    type_counts,
                    values='数量',
                    names='异常类型',
                    title='异常类型分布',
                    hole=0.4
                )
                fig_anom.update_layout(height=350)
                st.plotly_chart(fig_anom, use_container_width=True)

            with col_a2:
                st.markdown("### 📝 异常明细列表")
                st.dataframe(anomaly_df, use_container_width=True, hide_index=True, height=350)

            st.markdown("---")
            st.markdown("### 🔎 异常订单详情（关联出杯数据）")
            anomaly_rows = sorted(set(anomaly_df['行号'].tolist()))
            orig_idx = [r - 2 for r in anomaly_rows if r - 2 < len(df)]
            if orig_idx:
                anom_detail = df.iloc[orig_idx].copy()
                display_cols = [c for c in ['record_date', 'store_name', 'drink_name', 'barista_name',
                                           'order_minute', 'finish_minute', 'brew_duration', 'remake_flag', 'note']
                                if c in anom_detail.columns]
                st.dataframe(anom_detail[display_cols], use_container_width=True)
        else:
            st.success("✅ 未发现异常数据")

    with tab5:
        st.subheader("💡 备料调整建议")
        suggestions = get_prep_suggestions(df_filtered)

        if suggestions:
            for i, s in enumerate(suggestions, 1):
                with st.expander(f"**建议 {i}：优化 [{s['饮品']}] 备料方案**", expanded=True):
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.metric("📦 订单量", s['订单量'])
                    col_s2.metric("⏱️ 平均出杯时长", s['平均出杯时长(分钟)'] + " 分钟")
                    col_s3.metric("🔄 重做率", s['重做率'])

                    st.markdown(f"""
                    **🎯 问题点：** {s['问题点']}

                    **✅ 建议措施：**
                    - {s['建议']}
                    - 针对高重做率饮品，加强咖啡师制作标准培训
                    - 高峰时段（7-10点、12-14点、17-19点）增设备料人员
                    - 对出杯慢的饮品优化制作流程或引入半自动设备
                    """)
        else:
            st.info("当前数据暂未发现需要特别关注的备料优化点")

        st.markdown("---")
        st.markdown("### 📊 高峰时段分析")
        valid_hourly = df_filtered.dropna(subset=['order_min_parsed'])
        if len(valid_hourly) > 0:
            valid_hourly = valid_hourly.copy()
            valid_hourly['hour'] = (valid_hourly['order_min_parsed'] // 60).astype(int)
            hourly = valid_hourly.groupby('hour').agg(
                订单量=('hour', 'count'),
                平均出杯时长=('brew_duration', 'mean'),
                重做率=('is_remake', 'mean')
            ).reset_index()

            fig_hourly = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('各时段订单量分布', '各时段平均出杯时长 & 重做率')
            )
            colors = ['#FFE4B5' if (7 <= h <= 10) or (12 <= h <= 14) or (17 <= h <= 19) else '#DEB887'
                      for h in hourly['hour']]
            fig_hourly.add_trace(
                go.Bar(x=hourly['hour'], y=hourly['订单量'],
                       marker_color=colors, name='订单量'),
                row=1, col=1
            )
            fig_hourly.add_trace(
                go.Scatter(x=hourly['hour'], y=hourly['平均出杯时长'],
                           mode='lines+markers', name='平均出杯时长(分)',
                           line=dict(color='#8B4513', width=2)),
                row=2, col=1
            )
            fig_hourly.add_trace(
                go.Scatter(x=hourly['hour'], y=hourly['重做率']*100,
                           mode='lines+markers', name='重做率(%)',
                           line=dict(color='#FF4500', width=2),
                           yaxis='y2'),
                row=2, col=1
            )
            fig_hourly.update_layout(height=550, showlegend=True)
            st.plotly_chart(fig_hourly, use_container_width=True)

            st.caption("🟡 高亮色标注为典型高峰时段（早7-10 / 午12-14 / 晚17-19）")

    st.markdown("---")
    st.header("📥 数据下载")

    dcol1, dcol2 = st.columns(2)

    with dcol1:
        st.markdown("### 📋 筛选后明细 CSV")
        display_filtered = df_filtered.copy()
        if 'order_min_parsed' in display_filtered.columns:
            display_filtered['下单时间(HH:MM)'] = display_filtered['order_min_parsed'].apply(minute_to_hhmm)
        if 'finish_min_parsed' in display_filtered.columns:
            display_filtered['完成时间(HH:MM)'] = display_filtered['finish_min_parsed'].apply(minute_to_hhmm)
        if 'brew_duration' in display_filtered.columns:
            display_filtered['出杯时长(分钟)'] = display_filtered['brew_duration'].apply(
                lambda x: f"{x:.1f}" if pd.notna(x) else ""
            )
        if 'is_remake' in display_filtered.columns:
            display_filtered['是否重做'] = display_filtered['is_remake'].map({True: '是', False: '否'})

        csv_data = display_filtered.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⬇️ 下载筛选明细 CSV",
            data=csv_data,
            file_name=f"出杯明细_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
            use_container_width=True
        )

        if len(anomaly_df) > 0:
            anomaly_csv = anomaly_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="⬇️ 下载异常明细 CSV",
                data=anomaly_csv,
                file_name=f"异常明细_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )

    with dcol2:
        st.markdown("### 📄 HTML 分析报告")

        if len(valid_df) > 0:
            avg_duration = valid_df['brew_duration'].mean()
            remake_rate = df_filtered['is_remake'].mean() * 100
            total_orders = len(df_filtered)
            total_stores = df_filtered['store_name'].nunique()
        else:
            avg_duration = 0
            remake_rate = 0
            total_orders = 0
            total_stores = 0

        summary_metrics = {
            "总订单数": f"{total_orders} 单",
            "覆盖门店": f"{total_stores} 家",
            "平均出杯时长": f"{avg_duration:.1f} 分钟",
            "整体重做率": f"{remake_rate:.1f}%"
        }

        prep_suggestions = get_prep_suggestions(df_filtered)
        html_report = generate_html_report(df_filtered, df, anomaly_df, summary_metrics, prep_suggestions)
        st.download_button(
            label="⬇️ 下载 HTML 报告",
            data=html_report.encode('utf-8'),
            file_name=f"出杯效率报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime='text/html',
            use_container_width=True
        )

        with st.expander("📄 预览报告概览"):
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("总订单数", summary_metrics["总订单数"])
            pm2.metric("覆盖门店", summary_metrics["覆盖门店"])
            pm3.metric("平均出杯时长", summary_metrics["平均出杯时长"])
            pm4.metric("整体重做率", summary_metrics["整体重做率"])

if __name__ == "__main__":
    main()
