
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(layout="wide", page_title="Bank Marketing Prediction App")

@st.cache_resource
def load_model(model_path='best_model_output.pkl'):
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model ('{model_path}') not found. Please run the main script.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_pipeline = load_model()

original_columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
categorical_options = {'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'], 'marital': ['divorced', 'married', 'single', 'unknown'], 'education': ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'], 'default': ['no', 'unknown', 'yes'], 'housing': ['no', 'unknown', 'yes'], 'loan': ['no', 'unknown', 'yes'], 'contact': ['cellular', 'telephone'], 'month': ['apr', 'aug', 'dec', 'jul', 'jun', 'mar', 'may', 'nov', 'oct', 'sep'], 'day_of_week': ['fri', 'mon', 'thu', 'tue', 'wed'], 'poutcome': ['failure', 'nonexistent', 'success']}
numerical_ranges = {'age': (np.int64(18), np.int64(88)), 'duration': (np.int64(0), np.int64(3643)), 'campaign': (np.int64(1), np.int64(35)), 'pdays': (np.int64(0), np.int64(999)), 'previous': (np.int64(0), np.int64(6)), 'emp.var.rate': (np.float64(-3.4), np.float64(1.4)), 'cons.price.idx': (np.float64(92.201), np.float64(94.767)), 'cons.conf.idx': (np.float64(-50.8), np.float64(-26.9)), 'euribor3m': (np.float64(0.635), np.float64(5.045)), 'nr.employed': (np.float64(4963.6), np.float64(5228.1))}

st.title("🏦 Bank Term Deposit Subscription Prediction App")
st.markdown("Predicts whether a customer will subscribe to a term deposit product.")
st.markdown("---")

def get_user_input_sidebar():
    inputs = {}
    st.sidebar.header("Enter Customer Information")

    expander_sections = {
        "Personal Information": ['age', 'job', 'marital', 'education'],
        "Credit & Contact": ['default', 'housing', 'loan', 'contact', 'month', 'day_of_week'],
        "Campaign Information": ['duration', 'campaign', 'pdays', 'previous', 'poutcome'],
        "Economic Indicators": ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    }

    for section, cols in expander_sections.items():
        with st.sidebar.expander(section, expanded=(section=="Personal Information")):
            for col_name in cols:
                if col_name in original_columns:
                    if col_name in categorical_options:
                        options = categorical_options.get(col_name, ['unknown'])
                        if not options: options = ['unknown'] 
                        inputs[col_name] = st.selectbox(f"{col_name.replace('_',' ').title()}", options=options, index=0)

                    elif col_name in numerical_ranges:
                        min_val, max_val = numerical_ranges.get(col_name, (0,100))
                        default_val = (float(min_val) + float(max_val)) / 2.0
                        step_val = 1.0
                        if isinstance(min_val, float) or isinstance(max_val, float):
                            range_diff = float(max_val) - float(min_val)
                            if range_diff > 0:
                                if range_diff < 1: step_val = 0.01
                                elif range_diff < 20: step_val = 0.1

                        current_value = float(default_val)

                        if col_name == 'age' or col_name == 'campaign' or col_name == 'pdays' or col_name == 'previous' : 
                             inputs[col_name] = st.number_input(f"{col_name.replace('_',' ').title()}", 
                                                             min_value=int(round(min_val)), max_value=int(round(max_val)), 
                                                             value=int(round(current_value)), step=1)
                        elif isinstance(min_val, float) or isinstance(max_val, float) or (max_val - min_val < 30 and max_val != min_val): 
                             inputs[col_name] = st.slider(f"{col_name.replace('_',' ').title()}", 
                                                         min_value=float(min_val), max_value=float(max_val), 
                                                         value=current_value, step=step_val)
                        else: 
                            inputs[col_name] = st.number_input(f"{col_name.replace('_',' ').title()}", 
                                                            min_value=float(min_val), max_value=float(max_val), 
                                                            value=current_value, step=step_val if step_val > 0 else 0.1)

    final_input_data = {}
    for col in original_columns:
        final_input_data[col] = inputs.get(col) 

    input_df = pd.DataFrame([final_input_data])
    input_df = input_df[original_columns] 
    return input_df

if model_pipeline:

    user_input_df = get_user_input_sidebar()

    st.subheader("Entered Customer Information:")
    if not user_input_df.empty:
        display_df = user_input_df.copy().astype(str)
        st.dataframe(display_df)

    if st.sidebar.button("🔮 Predict", type="primary", use_container_width=True):
        if user_input_df.isnull().values.any():
            st.sidebar.error("Please fill in all fields. There are missing values.")
        else:
            try:
                prediction = model_pipeline.predict(user_input_df)
                prediction_proba = model_pipeline.predict_proba(user_input_df)
                st.markdown("---"); st.subheader("✨ Prediction Result ✨")
                col1_res, col2_res = st.columns(2)
                with col1_res:
                    if prediction[0] == 1: st.success("Customer is **EXPECTED TO SUBSCRIBE**.", icon="✅")
                    else: st.error("Customer is **NOT EXPECTED TO SUBSCRIBE**.", icon="❌")
                with col2_res:
                    proba_yes = prediction_proba[0][1]
                    st.metric(label="Subscription Probability (Yes)", value=f"{proba_yes*100:.2f}%")
                    st.progress(float(proba_yes))

                import altair as alt
                proba_data = pd.DataFrame({'Status': ['No', 'Yes'], 'Probability': prediction_proba[0]})
                chart = alt.Chart(proba_data).mark_bar().encode(
                    x=alt.X('Probability:Q', axis=alt.Axis(format='%')), y=alt.Y('Status:N', sort=None),
                    color=alt.Color('Status:N', legend=None, scale=alt.Scale(domain=['Yes', 'No'], range=['#2ECC71', '#E74C3C']))
                ).properties(title='Prediction Probabilities')
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.info("Enter the information in the left panel and click 'Predict' for a prediction.")
else:
    st.error(f"Model ('best_model_output.pkl') could not be loaded. Please run the main script.")

st.sidebar.markdown("---")
st.sidebar.markdown("##### ADA442 Project")
st.sidebar.markdown("- Anıl Metin")
st.sidebar.markdown("- Salih Melih Bağ")
st.sidebar.markdown("- İrem Şimşek")
st.sidebar.markdown("- Merve Nair")

st.markdown("---")
