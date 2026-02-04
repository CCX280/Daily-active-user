import streamlit as st
import joblib
import pandas as pd
import io
from datetime import date

# --- 1. 页面配置 ---
st.set_page_config(page_title="DAU diff", page_icon="📈")
st.title("📈 DAU diff")

# --- 2. 加载模型 ---
@st.cache_resource
def load_model():
    with st.spinner('正在搬运模型文件，请稍候...'):
        return joblib.load('dau_model_package.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ 模型文件没找到！请确认 'dau_model_package.pkl' 在同一个文件夹里。\n报错: {e}")
    st.stop()

# --- 3. 特征定义 ---
feature_names = [
    "week", 
    "is_holiday", 
    "is_workday", 
    "last_week_dau", 
    "yesterday_push", 
    "last_3days_ratio", 
    "is_in_holiday_time_front", 
    "is_in_holiday_time_behind", 
    "is_firstday_holiday", 
    "trend_ratio", 
    "month",
    "week_index"
]

# --- 4. 参数输入区 ---
st.sidebar.header("参数设置")
# 文件上传：支持 .xlsx
uploaded_file = st.sidebar.file_uploader("上传数据文件 (.xlsx)", type=["xlsx"])
uploaded_df = None
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_excel(uploaded_file)
        st.sidebar.success(f"已读取上传文件，形状: {uploaded_df.shape}")
        # 展示前几行作为预览
        st.sidebar.dataframe(uploaded_df.head())
    except Exception as e:
        st.sidebar.error(f"读取上传文件失败: {e}")
pick_date = st.sidebar.date_input("选择预测日期", value=date.today())

# 自动计算时间特征 (基准日 2023-01-01)
base_date = date(2023, 1, 1)
month = pick_date.month
week = pick_date.weekday() + 1
delta_days = (pick_date - base_date).days
week_index = (delta_days // 7) + 1 

st.sidebar.info(f"📅 选中日期是：自2023年1月1日以来的第 {week_index} 周")

# 其他输入
is_holiday = st.sidebar.selectbox("是否为节假日", [0, 1])
is_workday = st.sidebar.selectbox("是否为工作日", [0, 1])
last_week_dau = st.sidebar.number_input("上周 DAU", value=12000)
yesterday_push = st.sidebar.number_input("昨日 Push 量", value=5000)
last_3days_ratio = st.sidebar.number_input("近3日次留率均值", 0.0, 1.0, 0.2)
trend_ratio = st.sidebar.number_input("趋势系数 (Trend)", 0.0, 100.0, 0.98)

with st.sidebar.expander("更多节假日特征"):
    is_in_holiday_time_front = st.selectbox("假期前段", [0,1])
    is_in_holiday_time_behind = st.selectbox("假期后段", [0,1])
    is_firstday_holiday = st.selectbox("是否假期首日", [0, 1])

# 如果上传了 Excel，则尝试对每一行进行批量预测并展示
if uploaded_df is not None:
    st.header("📥 上传数据批量预测结果")
    df = uploaded_df.copy()
    # 检查必要特征列是否存在
    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        st.error(f"上传的 Excel 缺少必要特征列：{missing_cols}。请确保包含这些列，且列名与特征名匹配。")
    else:
        X = df[feature_names].copy()
        try:
            if isinstance(model, dict):
                xgb_preds = model['xgb_model'].predict(X)
                rf_preds = model['rf_model'].predict(X)
                weights_dict = model.get('weights', {"xgb": 0.7, "rf": 0.3})
                w_xgb = weights_dict.get('xgb', 0.7)
                w_rf = weights_dict.get('rf', 0.3)
                preds = (w_xgb * xgb_preds) + (w_rf * rf_preds)
                st.info(f"💡 融合详情: 使用权重 XGB={w_xgb}, RF={w_rf}")
            else:
                preds = model.predict(X)

            df['predicted_dau'] = pd.Series(preds).round().astype(int)
            st.success(f"已对上传表格的 {len(df)} 行进行预测")

            # 美化展示：数值千分位格式 + 高亮预测最大值
            num_cols = df.select_dtypes(include='number').columns.tolist()
            fmt = {c: '{:,.0f}' for c in num_cols}
            styled = df.style.format(fmt).highlight_max(subset=['predicted_dau'], color='#b6e3a8')
            st.write(styled)

            # 提供下载带预测结果的 Excel
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            towrite.seek(0)
            st.download_button("下载带预测结果的 Excel", data=towrite, file_name="predicted_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        except Exception as e:
            st.error(f"批量预测失败: {e}")

# --- 5. 核心预测 (精准匹配版) ---
if st.button("🚀 开始预测"):
    input_data = pd.DataFrame(
        [[
            week, is_holiday, is_workday, last_week_dau, yesterday_push,
            last_3days_ratio, is_in_holiday_time_front, is_in_holiday_time_behind,
            is_firstday_holiday, trend_ratio, month, week_index
        ]], 
        columns=feature_names
    )

    try:
        if isinstance(model, dict):
            # 1. 预测
            xgb_pred = model['xgb_model'].predict(input_data)[0]
            rf_pred = model['rf_model'].predict(input_data)[0]
            
            # 2. 读取权重 (修正点：用键名读取，而不是索引)
            weights_dict = model.get("weights") # 这是一个字典 {"xgb": 0.7, "rf": 0.3}
            w_xgb = weights_dict["xgb"]
            w_rf = weights_dict["rf"]
            
            # 3. 融合
            final_pred = (w_xgb * xgb_pred) + (w_rf * rf_pred)
            
            st.info(f"💡 融合详情: XGB({int(xgb_pred)}) x {w_xgb} + RF({int(rf_pred)}) x {w_rf}")
        else:
            final_pred = model.predict(input_data)[0]

        st.success(f"🔮 最终预测结果：{int(final_pred):,}")

    except Exception as e:
        st.error(f"❌ 运行出错: {e}")
