import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- 1. 页面配置 ---
st.set_page_config(page_title="DAU 预测", page_icon="📈")
st.title("📈 DAU 波动预测 (最终版)")

# --- 2. 加载已经封装好的“完全体”模型 ---
@st.cache_resource
def load_model():
    # 既然是完整的 pkl，加载出来就是一个能直接 predict 的对象
    return joblib.load('dau_model_package.pkl')

model = load_model()

# --- 3. 特征定义 (保持和训练时一致的顺序) ---
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
]

# --- 4. 参数输入区 ---
st.sidebar.header("参数设置")
pick_date = st.sidebar.date_input("选择预测日期")
month = pick_date.month
week = pick_date.weekday() + 1

is_holiday = st.sidebar.selectbox("是否为节假日", [0, 1])
is_workday = st.sidebar.selectbox("是否为工作日", [0, 1])

# 数值输入
last_week_dau = st.sidebar.number_input("上周 DAU", value=12000)
yesterday_push = st.sidebar.number_input("昨日 Push 量", value=5000)
last_3days_ratio = st.sidebar.number_input("近3日次留率均值", 0.0, 1.0, 0.2)
trend_ratio = st.sidebar.number_input("趋势系数 (Trend)", 0.0, 100.0, 0.98)

# 节假日细节
with st.sidebar.expander("更多节假日特征"):
    is_in_holiday_time_front = st.selectbox("假期前段", [0,1,2,3,4, 5])
    is_in_holiday_time_behind = st.selectbox("假期后段", [0,1, 2,3,4,5])
    is_firstday_holiday = st.selectbox("是否假期首日", [0, 1])

# --- 5. 核心预测 (极简版) ---
# --- 5. 核心预测逻辑 (适配字典包) ---
if st.button("🚀 开始预测"):
    # 1. 构造数据
    input_data = pd.DataFrame(
        [[
            week, is_holiday, is_workday, last_week_dau, yesterday_push,
            last_3days_ratio, is_in_holiday_time_front, is_in_holiday_time_behind,
            is_firstday_holiday, trend_ratio, month
        ]], 
        columns=feature_names
    )

    try:
        # 核心修复在这里：判断它是不是一个字典
        if isinstance(model, dict):
            # 如果是字典，说明要把里面的两个模型拿出来分别预测
            xgb_pred = model['xgb'].predict(input_data)[0]
            rf_pred = model['rf'].predict(input_data)[0]
            
            # 手动融合 (0.7 * XGB + 0.3 * RF)
            # 这里的权重最好也从包里读，或者写死
            w = model.get('weights', [0.7, 0.3])
            final_pred = (w[0] * xgb_pred) + (w[1] * rf_pred)
            
            st.info(f"💡 融合详情: XGB({int(xgb_pred)}) x {w[0]} + RF({int(rf_pred)}) x {w[1]}")
        else:
            # 如果它真的是个单一模型对象（以后你可能会存这种），直接预测
            final_pred = model.predict(input_data)[0]

        st.success(f"🔮 最终预测结果：{int(final_pred):,}")

    except Exception as e:
        st.error(f"❌ 运行出错: {e}")
        # 如果出错，打印一下 model 到底是个啥，方便调试
        st.write("调试信息：你的模型类型是", type(model))
        if isinstance(model, dict):
            st.write("字典里的钥匙有：", model.keys())

