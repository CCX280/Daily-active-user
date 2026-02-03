import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. 页面配置 ---
st.set_page_config(page_title="DAU 预测", page_icon="📈")
st.title("📈 DAU 波动预测 (最终版)")

# --- 2. 加载模型 (加了加载提示，防止你以为它卡死) ---
@st.cache_resource
def load_model():
    with st.spinner('正在搬运模型文件，请稍候...'):
        return joblib.load('dau_model_package.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ 模型文件没找到！请确认 'dau_model_package.pkl' 在同一个文件夹里。\n报错: {e}")
    st.stop() # 没模型就别往下跑了

# --- 3. 特征定义 ---
feature_names = [
    "week", "is_holiday", "is_workday", "last_week_dau", 
    "yesterday_push", "last_3days_ratio", "is_in_holiday_time_front", 
    "is_in_holiday_time_behind", "is_firstday_holiday", "trend_ratio", "month"
]

# --- 4. 参数输入区 ---
st.sidebar.header("参数设置")
pick_date = st.sidebar.date_input("选择预测日期")
month = pick_date.month
week = pick_date.weekday() + 1

is_holiday = st.sidebar.selectbox("是否为节假日", [0, 1])
is_workday = st.sidebar.selectbox("是否为工作日", [0, 1])
last_week_dau = st.sidebar.number_input("上周 DAU", value=12000)
yesterday_push = st.sidebar.number_input("昨日 Push 量", value=5000)
last_3days_ratio = st.sidebar.number_input("近3日次留率均值", 0.0, 1.0, 0.2)
trend_ratio = st.sidebar.number_input("趋势系数 (Trend)", 0.0, 100.0, 0.98)

with st.sidebar.expander("更多节假日特征"):
    is_in_holiday_time_front = st.selectbox("假期前段", [0,1，2,3,4,5])
    is_in_holiday_time_behind = st.selectbox("假期后段", [0,1,2,3,4,5])
    is_firstday_holiday = st.selectbox("是否假期首日", [0, 1])

# --- 5. 核心预测 (KeyError 修复版) ---
if st.button("🚀 开始预测"):
    # 构造数据
    input_data = pd.DataFrame(
        [[
            week, is_holiday, is_workday, last_week_dau, yesterday_push,
            last_3days_ratio, is_in_holiday_time_front, is_in_holiday_time_behind,
            is_firstday_holiday, trend_ratio, month
        ]], 
        columns=feature_names
    )

    try:
        # 判断是否为字典包
        if isinstance(model, dict):
            # 修复点：这里改成了 'xgb_model' 和 'rf_model'
            xgb_pred = model['xgb_model'].predict(input_data)[0]
            rf_pred = model['rf_model'].predict(input_data)[0]
            
            w = model.get('weights', [0.7, 0.3])
            final_pred = (w[0] * xgb_pred) + (w[1] * rf_pred)
            
            st.info(f"💡 融合详情: XGB({int(xgb_pred)}) x {w[0]} + RF({int(rf_pred)}) x {w[1]}")
        else:
            final_pred = model.predict(input_data)[0]

        st.success(f"🔮 最终预测结果：{int(final_pred):,}")

    except Exception as e:
        st.error(f"❌ 运行出错: {e}")
        st.write("调试信息：你的模型里的钥匙是：", model.keys() if isinstance(model, dict) else "不是字典")