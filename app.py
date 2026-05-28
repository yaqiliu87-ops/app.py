import io

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="统计分析工作台",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def make_demo_data(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=180, freq="D")
    regions = np.array(["华东", "华北", "华南", "西南", "西北"])
    channels = np.array(["线上", "门店", "代理", "企业客户"])
    products = np.array(["基础版", "专业版", "旗舰版"])

    data = pd.DataFrame(
        {
            "日期": rng.choice(dates, rows),
            "地区": rng.choice(regions, rows, p=[0.28, 0.2, 0.24, 0.16, 0.12]),
            "渠道": rng.choice(channels, rows, p=[0.42, 0.26, 0.2, 0.12]),
            "产品": rng.choice(products, rows, p=[0.46, 0.34, 0.2]),
            "订单数": rng.poisson(9, rows) + 1,
            "客单价": rng.normal(620, 145, rows).clip(120, 1300).round(2),
            "转化率": rng.beta(8, 38, rows).round(4),
            "满意度": rng.normal(4.25, 0.42, rows).clip(1, 5).round(2),
        }
    )
    product_multiplier = data["产品"].map({"基础版": 0.86, "专业版": 1.08, "旗舰版": 1.36})
    channel_multiplier = data["渠道"].map({"线上": 1.04, "门店": 0.96, "代理": 0.92, "企业客户": 1.18})
    data["销售额"] = (data["订单数"] * data["客单价"] * product_multiplier * channel_multiplier).round(2)
    return data.sort_values("日期").reset_index(drop=True)


def load_csv(uploaded_file) -> pd.DataFrame:
    content = uploaded_file.read()
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(content))


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in result.columns:
        if result[column].dtype == "object":
            parsed = pd.to_datetime(result[column], errors="coerce")
            if parsed.notna().mean() > 0.75:
                result[column] = parsed
    return result


def filtered_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("数据与筛选")
    uploaded_file = st.sidebar.file_uploader("上传 CSV 数据", type=["csv"])
    st.sidebar.caption("未上传时使用内置示例数据。")

    if uploaded_file:
        df = coerce_dates(load_csv(uploaded_file))

    date_columns = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    category_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if date_columns:
        date_column = st.sidebar.selectbox("日期字段", date_columns)
        min_date = df[date_column].min().date()
        max_date = df[date_column].max().date()
        selected_range = st.sidebar.date_input("日期范围", value=(min_date, max_date))
        if len(selected_range) == 2:
            start, end = selected_range
            df = df[(df[date_column].dt.date >= start) & (df[date_column].dt.date <= end)]

    for column in category_columns[:5]:
        values = sorted(df[column].dropna().astype(str).unique().tolist())
        if 1 < len(values) <= 30:
            selected = st.sidebar.multiselect(column, values, default=values)
            df = df[df[column].astype(str).isin(selected)]

    return df


def metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


base_df = make_demo_data()
df = filtered_data(base_df)

st.title("统计分析工作台")
st.caption("上传 CSV 或使用示例数据，快速查看核心指标、分布、趋势和字段关系。")

if df.empty:
    st.warning("当前筛选条件下没有数据。请调整筛选范围。")
    st.stop()

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
category_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
date_columns = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

left, middle, right, extra = st.columns(4)
with left:
    metric_card("记录数", f"{len(df):,}")
with middle:
    metric_card("字段数", f"{df.shape[1]:,}")
with right:
    missing_rate = df.isna().mean().mean()
    metric_card("整体缺失率", f"{missing_rate:.1%}")
with extra:
    if numeric_columns:
        metric_card("数值字段", f"{len(numeric_columns):,}")
    else:
        metric_card("数值字段", "0")

tab_overview, tab_charts, tab_relation, tab_data = st.tabs(
    ["概览", "图表", "关系", "数据"]
)

with tab_overview:
    st.subheader("统计摘要")
    if numeric_columns:
        st.dataframe(
            df[numeric_columns].describe().T.round(3),
            use_container_width=True,
        )
    else:
        st.info("没有检测到数值字段。")

    st.subheader("缺失值")
    missing = (
        df.isna()
        .sum()
        .rename("缺失数量")
        .to_frame()
        .assign(缺失比例=lambda x: x["缺失数量"] / len(df))
        .sort_values("缺失数量", ascending=False)
    )
    st.dataframe(missing, use_container_width=True)

with tab_charts:
    chart_left, chart_right = st.columns([1, 1])
    with chart_left:
        st.subheader("数值分布")
        if numeric_columns:
            metric = st.selectbox("选择数值字段", numeric_columns)
            bins = st.slider("分箱数量", 5, 80, 30)
            counts, edges = np.histogram(df[metric].dropna(), bins=bins)
            hist_df = pd.DataFrame(
                {
                    "区间": [f"{edges[i]:.2f} - {edges[i + 1]:.2f}" for i in range(len(edges) - 1)],
                    "数量": counts,
                }
            )
            st.bar_chart(hist_df.set_index("区间"))
        else:
            st.info("没有可绘制分布的数值字段。")

    with chart_right:
        st.subheader("分类对比")
        if category_columns and numeric_columns:
            group_col = st.selectbox("分组字段", category_columns)
            value_col = st.selectbox("汇总字段", numeric_columns, index=min(1, len(numeric_columns) - 1))
            agg_method = st.segmented_control(
                "汇总方式",
                ["总和", "平均", "中位数", "计数"],
                default="总和",
            )
            agg_map = {
                "总和": "sum",
                "平均": "mean",
                "中位数": "median",
                "计数": "count",
            }
            grouped = (
                df.groupby(group_col, dropna=False)[value_col]
                .agg(agg_map[agg_method])
                .sort_values(ascending=False)
                .head(20)
            )
            st.bar_chart(grouped)
        else:
            st.info("需要至少一个分类字段和一个数值字段。")

    if date_columns and numeric_columns:
        st.subheader("时间趋势")
        date_col = st.selectbox("趋势日期字段", date_columns)
        value_col = st.selectbox("趋势指标", numeric_columns, index=len(numeric_columns) - 1)
        trend = (
            df[[date_col, value_col]]
            .dropna()
            .set_index(date_col)
            .resample("D")[value_col]
            .sum()
        )
        st.line_chart(trend)

with tab_relation:
    st.subheader("字段关系")
    if len(numeric_columns) >= 2:
        x_col, y_col = st.columns(2)
        with x_col:
            x_axis = st.selectbox("X 轴", numeric_columns, index=0)
        with y_col:
            y_axis = st.selectbox("Y 轴", numeric_columns, index=min(1, len(numeric_columns) - 1))
        st.scatter_chart(df[[x_axis, y_axis]].dropna(), x=x_axis, y=y_axis)

        st.subheader("相关性矩阵")
        corr = df[numeric_columns].corr(numeric_only=True).round(3)
        st.dataframe(corr, use_container_width=True)
    else:
        st.info("至少需要两个数值字段才能分析相关性。")

with tab_data:
    st.subheader("数据预览")
    st.dataframe(df, use_container_width=True, height=420)

    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "下载当前筛选数据",
        data=csv,
        file_name="filtered_statistics_data.csv",
        mime="text/csv",
    )
