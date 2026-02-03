import streamlit as st
import joblib
import pandas as pd
import numpy as np
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