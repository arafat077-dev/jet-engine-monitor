import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="JetGuard AI — Engine Health Monitor",
                   page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0A0E1A; }
section[data-testid="stSidebar"] { background: #0D1117 !important; border-right: 1px solid #1E2A3A; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stTabs [data-baseweb="tab-list"] { background: #0D1117; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid #1E2A3A; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 8px; color: #8892A4; font-weight: 500; font-size: 0.9rem; padding: 8px 18px; border: none; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1A56DB, #0EA5E9) !important; color: white !important; }
.stButton > button { background: linear-gradient(135deg, #1A56DB, #0EA5E9); color: white; border: none; border-radius: 10px; padding: 10px 24px; font-weight: 600; font-size: 0.95rem; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-2px); }
.stDownloadButton > button { background: linear-gradient(135deg, #1A56DB, #0EA5E9); color: white; border: none; border-radius: 10px; font-weight: 600; }
[data-testid="stMetric"] { background: #0D1117; border: 1px solid #1E2A3A; border-radius: 12px; padding: 16px 20px; }
[data-testid="stMetricLabel"] { color: #8892A4 !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #E2E8F0 !important; font-size: 1.4rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }
.stSelectbox > div > div, .stNumberInput > div > div { background: #0D1117 !important; border: 1px solid #1E2A3A !important; border-radius: 8px !important; color: #E2E8F0 !important; }
[data-testid="stFileUploader"] { background: #0D1117; border: 2px dashed #2A3A52; border-radius: 12px; padding: 16px; }
[data-testid="stFileUploader"] label { color: #8892A4 !important; }
[data-testid="stFileUploaderDropzone"] { background: #111827; border-radius: 8px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0E1A; }
::-webkit-scrollbar-thumb { background: #1E2A3A; border-radius: 3px; }
.kpi-card { background: linear-gradient(135deg, #0D1117, #111827); border: 1px solid #1E2A3A; border-radius: 16px; padding: 18px 20px; text-align: center; position: relative; overflow: hidden; }
.kpi-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #1A56DB, #0EA5E9); }
.kpi-value { font-size: 2rem; font-weight: 800; margin: 6px 0 2px; }
.section-header { font-size: 1.1rem; font-weight: 700; color: #E2E8F0; margin: 22px 0 10px; padding-bottom: 8px; border-bottom: 1px solid #1E2A3A; }
.helptext { color: #8892A4; font-size: 0.85rem; margin: 0 0 14px 0; line-height: 1.5; }
.alert-box { border-radius: 12px; padding: 16px 20px; margin: 14px 0; border-left: 4px solid; }
.alert-healthy  { background: rgba(16,185,129,0.08); border-color: #10B981; color: #6EE7B7; }
.alert-monitor  { background: rgba(245,158,11,0.08); border-color: #F59E0B; color: #FCD34D; }
.alert-warning  { background: rgba(249,115,22,0.08); border-color: #F97316; color: #FDBA74; }
.alert-critical { background: rgba(239,68,68,0.08);  border-color: #EF4444; color: #FCA5A5; }
.criteria-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; border-radius: 8px; margin: 5px 0; background: #0D1117; border: 1px solid #1E2A3A; }
.sidebar-metric { background: #111827; border: 1px solid #1E2A3A; border-radius: 10px; padding: 11px 14px; margin: 6px 0; }
.sidebar-metric-label { font-size: 0.7rem; color: #8892A4; text-transform: uppercase; letter-spacing: 0.04em; }
.sidebar-metric-value { font-size: 1.05rem; font-weight: 700; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    iso = joblib.load('isolation_forest_v3.pkl')
    reg = joblib.load('rul_regressor.pkl')
    sc_iso = joblib.load('scaler_anomaly.pkl')
    sc_reg = joblib.load('scaler_regression.pkl')
    with open('model_config.json') as f:
        cfg = json.load(f)
    return iso, reg, sc_iso, sc_reg, cfg

try:
    iso_model, reg_model, sc_iso, sc_reg, config = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

STATUS_COLORS = {'HEALTHY':'#10B981','MONITOR':'#F59E0B','WARNING':'#F97316','CRITICAL':'#EF4444'}
STATUS_LABELS = {'HEALTHY':'All Clear','MONITOR':'Watch Closely','WARNING':'Attention Needed','CRITICAL':'Immediate Action'}

def base_layout():
    return dict(template='plotly_dark', paper_bgcolor='rgba(13,17,23,0)',
                plot_bgcolor='rgba(13,17,23,0)',
                font=dict(family='Inter', color='#8892A4', size=12),
                margin=dict(l=40, r=20, t=50, b=40))

def engineer_features(df_input):
    base = config['base_features']
    df = df_input.copy().sort_values(['engine_id','cycle']).reset_index(drop=True)
    for col in base:
        if col not in df.columns:
            df[col] = 0.0
    roll5=[]; roll10=[]; lag1=[]; delta=[]
    for col in base:
        df[f'{col}_roll5']  = df.groupby('engine_id')[col].transform(lambda x: x.rolling(5,  min_periods=1).mean())
        df[f'{col}_roll10'] = df.groupby('engine_id')[col].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df[f'{col}_lag1']   = df.groupby('engine_id')[col].transform(lambda x: x.shift(1))
        df[f'{col}_delta']  = df.groupby('engine_id')[col].transform(lambda x: x.diff())
        roll5.append(f'{col}_roll5'); roll10.append(f'{col}_roll10')
        lag1.append(f'{col}_lag1'); delta.append(f'{col}_delta')
    df = df.fillna(0)
    return df, base + roll5 + roll10 + lag1 + delta

def predict(df_eng, feats):
    X = df_eng[feats].values
    df_eng['anomaly_flag']  = (iso_model.predict(sc_iso.transform(X)) == -1).astype(int)
    df_eng['anomaly_score'] = iso_model.decision_function(sc_iso.transform(X))
    df_eng['RUL_predicted'] = np.clip(reg_model.predict(sc_reg.transform(X)), 0, config['rul_cap'])
    def status(row):
        if row['anomaly_flag']==1 and row['RUL_predicted']<=30: return 'CRITICAL'
        if row['anomaly_flag']==1 or  row['RUL_predicted']<=30: return 'WARNING'
        if row['RUL_predicted']<=60: return 'MONITOR'
        return 'HEALTHY'
    df_eng['status'] = df_eng.apply(status, axis=1)
    return df_eng

with st.sidebar:
    st.markdown("""<div style="padding:16px 0 14px;border-bottom:1px solid #1E2A3A;margin-bottom:14px;">
        <div style="font-size:1.4rem;font-weight:800;color:#E2E8F0;">JetGuard <span style="color:#1A56DB;">AI</span></div>
        <div style="font-size:0.72rem;color:#8892A4;margin-top:2px;">Predictive Maintenance System</div></div>""", unsafe_allow_html=True)
    if models_loaded:
        m = config['metrics']
        st.markdown('<div style="font-size:0.7rem;color:#8892A4;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Live Model Performance</div>', unsafe_allow_html=True)
        for label,val,color in [("Detection Lead Time",f"{m['anomaly_lead_time']} cycles","#1A56DB"),
                                 ("False Alarm Rate",f"{m['anomaly_fpr']}%","#10B981"),
                                 ("RUL Accuracy (MAE)",f"{m['rul_mae']} cycles","#F59E0B"),
                                 ("Near-Failure MAE",f"{m['rul_mae_critical']} cycles","#EF4444"),
                                 ("R\u00b2 Score",f"{m['rul_r2']}","#8B5CF6")]:
            st.markdown(f'<div class="sidebar-metric"><div class="sidebar-metric-label">{label}</div><div class="sidebar-metric-value" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:14px"></div><div style="font-size:0.7rem;color:#8892A4;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Status Guide</div>', unsafe_allow_html=True)
        for s,c in STATUS_COLORS.items():
            st.markdown(f'<div style="padding:6px 10px;margin:3px 0;border-radius:6px;background:rgba(255,255,255,0.03);border-left:3px solid {c};font-size:0.8rem;color:#E2E8F0;"><b style="color:{c}">{s}</b> <span style="color:#8892A4;font-size:0.72rem">— {STATUS_LABELS[s]}</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:14px"></div><div style="padding:12px;background:#111827;border-radius:10px;border:1px solid #1E2A3A;font-size:0.74rem;color:#8892A4;line-height:1.6;"><div style="color:#E2E8F0;font-weight:600;margin-bottom:4px;">Group 2 — Sejong University</div>Basic Design Course<br>Prof. Abolghasem Sadeghi-Niaraki<br><span style="color:#1A56DB;">NASA C-MAPSS FD001</span></div>', unsafe_allow_html=True)

st.markdown("""<div style="background:linear-gradient(135deg,#0D1B35 0%,#0A0E1A 100%);border:1px solid #1E2A3A;border-radius:20px;padding:28px 36px;margin-bottom:22px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;background:radial-gradient(circle,rgba(26,86,219,0.15),transparent 70%);border-radius:50%;"></div>
    <div style="position:relative;z-index:1;"><div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
    <div style="background:linear-gradient(135deg,#1A56DB,#0EA5E9);width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.4rem;">&#9992;</div>
    <div><div style="font-size:1.7rem;font-weight:800;color:#E2E8F0;line-height:1.1;">JetGuard <span style="background:linear-gradient(135deg,#1A56DB,#0EA5E9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">AI</span></div>
    <div style="font-size:0.82rem;color:#8892A4;margin-top:2px;">Jet Engine Predictive Maintenance System</div></div></div>
    <div style="display:flex;gap:10px;margin-top:14px;flex-wrap:wrap;">
    <div style="background:rgba(26,86,219,0.15);border:1px solid rgba(26,86,219,0.3);border-radius:20px;padding:4px 14px;font-size:0.76rem;color:#93C5FD;">Isolation Forest V3</div>
    <div style="background:rgba(14,165,233,0.15);border:1px solid rgba(14,165,233,0.3);border-radius:20px;padding:4px 14px;font-size:0.76rem;color:#7DD3FC;">Gradient Boosting RUL</div>
    <div style="background:rgba(16,185,129,0.15);border:1px solid rgba(16,185,129,0.3);border-radius:20px;padding:4px 14px;font-size:0.76rem;color:#6EE7B7;">NASA C-MAPSS FD001</div>
    <div style="background:rgba(139,92,246,0.15);border:1px solid rgba(139,92,246,0.3);border-radius:20px;padding:4px 14px;font-size:0.76rem;color:#C4B5FD;">45 Features</div></div></div></div>""", unsafe_allow_html=True)

if not models_loaded:
    st.error("Model files not found. Place all 5 files (isolation_forest_v3.pkl, rul_regressor.pkl, scaler_anomaly.pkl, scaler_regression.pkl, model_config.json) in the same folder as app.py")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📁  Upload CSV","🔧  Manual Input","📊  Model Performance","🚀  Project Journey"])

with tab1:
    st.markdown('<div class="section-header">Upload Engine Sensor Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a CSV with columns: engine_id, cycle, and the 9 sensors (sensor_2, 3, 4, 7, 11, 12, 17, 20, 21)",
        type=['csv'], key="csv_upload")
    with st.expander("View expected CSV format"):
        sample = pd.DataFrame({'engine_id':[1,1,2],'cycle':[1,2,1],'sensor_2':[641.8,642.1,640.5],
            'sensor_3':[1589.0,1588.2,1590.1],'sensor_4':[1400.6,1399.8,1401.2],'sensor_7':[554.4,554.1,554.6],
            'sensor_11':[47.47,47.51,47.30],'sensor_12':[522.4,522.1,522.6],'sensor_17':[392,392,391],
            'sensor_20':[39.06,39.02,39.10],'sensor_21':[23.42,23.40,23.45]})
        st.dataframe(sample, use_container_width=True, hide_index=True)
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        n_engines = df_raw['engine_id'].nunique(); n_records = len(df_raw)
        st.success(f"Loaded {n_records:,} records from {n_engines} engines")
        with st.spinner("Running dual-model analysis..."):
            df_eng, feats = engineer_features(df_raw)
            df_result = predict(df_eng, feats)
        st.markdown('<div class="section-header">Fleet Health Overview</div>', unsafe_allow_html=True)
        engine_last = df_result.sort_values('cycle').groupby('engine_id').last().reset_index()
        status_dist = engine_last['status'].value_counts()
        c1,c2,c3,c4 = st.columns(4)
        for col,s in zip([c1,c2,c3,c4],['HEALTHY','MONITOR','WARNING','CRITICAL']):
            count = int(status_dist.get(s,0)); pct = count/n_engines*100 if n_engines else 0
            color = STATUS_COLORS[s]
            with col:
                st.markdown(f'<div class="kpi-card"><div style="font-size:0.7rem;color:#8892A4;text-transform:uppercase;letter-spacing:.06em;">{s}</div><div class="kpi-value" style="color:{color}">{count}</div><div style="font-size:0.74rem;color:{color};opacity:0.8">{pct:.0f}% of fleet</div><div style="font-size:0.7rem;color:#8892A4;margin-top:3px">{STATUS_LABELS[s]}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Fleet Distribution & Engine Table</div>', unsafe_allow_html=True)
        col_d, col_t = st.columns([1,1.6])
        with col_d:
            labels = list(status_dist.index); values = [int(v) for v in status_dist.values]
            colors = [STATUS_COLORS[l] for l in labels]
            fig_d = go.Figure(go.Pie(labels=labels, values=values, hole=0.62,
                    marker=dict(colors=colors, line=dict(color='#0A0E1A', width=3)),
                    textfont=dict(size=12, color='white'),
                    hovertemplate='<b>%{label}</b><br>%{value} engines<br>%{percent}<extra></extra>'))
            fig_d.add_annotation(text=f"<b>{n_engines}</b><br>Engines", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color='#E2E8F0'))
            ld = base_layout(); ld['height']=280; ld['showlegend']=True
            ld['legend']=dict(orientation='h', y=-0.1, font=dict(color='#E2E8F0', size=10))
            fig_d.update_layout(**ld)
            st.plotly_chart(fig_d, use_container_width=True)
        with col_t:
            tbl = engine_last[['engine_id','RUL_predicted','anomaly_flag','status']].copy()
            tbl['RUL_predicted'] = tbl['RUL_predicted'].round(1)
            tbl['anomaly_flag'] = tbl['anomaly_flag'].map({0:'Normal',1:'Anomaly'})
            tbl.columns = ['Engine ID','Predicted RUL','Anomaly','Health Status']
            st.dataframe(tbl, use_container_width=True, height=280, hide_index=True)
        st.markdown('<div class="section-header">Engine Deep Dive</div>', unsafe_allow_html=True)
        sel = st.selectbox("Select engine to inspect:", sorted(df_result['engine_id'].unique()))
        ed = df_result[df_result['engine_id']==sel].sort_values('cycle')
        cur_status = ed.iloc[-1]['status']; cur_rul = ed.iloc[-1]['RUL_predicted']; anom_count = int(ed['anomaly_flag'].sum())
        st.markdown(f'<div class="alert-box alert-{cur_status.lower()}"><b style="font-size:1rem">Engine #{sel} — {cur_status}</b><span style="margin-left:14px;font-size:0.88rem">Predicted RUL: <b>{cur_rul:.1f} cycles</b> &nbsp;|&nbsp; Anomalies: <b>{anom_count}</b> of {len(ed)} records</span></div>', unsafe_allow_html=True)
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Health Status", cur_status); m2.metric("Predicted RUL", f"{cur_rul:.1f} cy")
        m3.metric("Anomalies", str(anom_count)); m4.metric("Total Cycles", str(len(ed)))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['RUL Prediction Track','Anomaly Score Track'], vertical_spacing=0.12, row_heights=[0.62,0.38])
        if 'RUL' in ed.columns:
            fig.add_trace(go.Scatter(x=ed['cycle'], y=ed['RUL'], name='Actual RUL', line=dict(color='#60A5FA', width=2), mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ed['cycle'], y=ed['RUL_predicted'], name='Predicted RUL', line=dict(color='#34D399', width=2, dash='dash'), mode='lines'), row=1, col=1)
        ap = ed[ed['anomaly_flag']==1]
        if len(ap):
            fig.add_trace(go.Scatter(x=ap['cycle'], y=ap['RUL_predicted'], name='Anomaly', mode='markers', marker=dict(color='#EF4444', size=7, symbol='x')), row=1, col=1)
        fig.add_hline(y=30, line=dict(color='#F59E0B', dash='dot', width=1.5), row=1, col=1)
        fig.add_trace(go.Scatter(x=ed['cycle'], y=ed['anomaly_score'], name='Anomaly Score', line=dict(color='#F59E0B', width=1.5), fill='tozeroy', fillcolor='rgba(245,158,11,0.08)', mode='lines'), row=2, col=1)
        fig.add_hline(y=0, line=dict(color='#EF4444', dash='dash', width=1), row=2, col=1)
        le = base_layout(); le['height']=470; le['showlegend']=True
        le['legend']=dict(orientation='h', y=1.08, bgcolor='rgba(0,0,0,0)')
        fig.update_layout(**le)
        fig.update_xaxes(gridcolor='#1E2A3A'); fig.update_yaxes(gridcolor='#1E2A3A')
        st.plotly_chart(fig, use_container_width=True)
        csv_out = df_result[['engine_id','cycle','RUL_predicted','anomaly_flag','anomaly_score','status']].to_csv(index=False)
        st.download_button("⬇️ Download Full Results (CSV)", data=csv_out, file_name="engine_health_results.csv", mime="text/csv")

with tab2:
    st.markdown('<div class="section-header">Manual Sensor Input</div>', unsafe_allow_html=True)
    st.markdown('<p class="helptext">Enter sensor readings to get an instant engine health assessment.</p>', unsafe_allow_html=True)
    ci,cc = st.columns(2)
    eng_id = ci.number_input("Engine ID", min_value=1, value=1, step=1)
    cycle = cc.number_input("Current Cycle", min_value=1, value=150, step=1)
    st.markdown('<div class="section-header">Sensor Readings</div>', unsafe_allow_html=True)
    sensors = [('sensor_2',641.82,518.67,644.0,"Fan Inlet Temp"),('sensor_3',1589.7,1400.0,1616.0,"LPC Outlet Temp"),
        ('sensor_4',1400.6,1100.0,1490.0,"HPC Outlet Temp"),('sensor_7',554.36,521.0,556.0,"HPC Outlet Pressure"),
        ('sensor_11',47.47,43.0,48.5,"Static Pressure"),('sensor_12',522.42,388.0,523.0,"Fuel Flow Ratio"),
        ('sensor_17',392.0,388.0,400.0,"Bypass Ratio"),('sensor_20',39.06,38.0,39.5,"Bleed Enthalpy"),
        ('sensor_21',23.42,23.0,23.6,"HPT Cool Air Flow")]
    vals = {}
    for row in [sensors[i:i+3] for i in range(0,len(sensors),3)]:
        cols = st.columns(3)
        for col,(s,dv,mn,mx,label) in zip(cols,row):
            with col:
                vals[s] = st.slider(f"{s} — {label}", float(mn), float(mx), float(dv), 0.01, format="%.2f")
    if st.button("🔍 Analyse Engine Health", type="primary", use_container_width=True):
        df_m = pd.DataFrame([dict(engine_id=eng_id, cycle=c, **vals) for c in range(max(1,cycle-14), cycle+1)])
        with st.spinner("Analysing..."):
            de, fm = engineer_features(df_m)
            dr = predict(de, fm)
        latest = dr.iloc[-1]
        status = latest['status']; pred_rul = latest['RUL_predicted']; anom = latest['anomaly_flag']; score = latest['anomaly_score']
        color = STATUS_COLORS[status]
        st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="alert-box alert-{status.lower()}"><div style="font-weight:800;font-size:1.2rem;margin-bottom:4px;">Engine #{eng_id} — {status}</div><div style="font-size:0.92rem">Predicted RUL: <b>{pred_rul:.1f} cycles</b> &nbsp;|&nbsp; Detection: <b>{"Anomaly" if anom else "Normal"}</b></div></div>', unsafe_allow_html=True)
        r1,r2,r3 = st.columns(3)
        r1.metric("Predicted RUL", f"{pred_rul:.1f} cy", "Safe" if pred_rul>60 else ("Caution" if pred_rul>30 else "Critical"))
        r2.metric("Anomaly", "Detected" if anom else "Normal"); r3.metric("Anomaly Score", f"{score:.4f}")
        recs = {'HEALTHY':"Engine operating within normal parameters. Continue scheduled monitoring.",
            'MONITOR':"Approaching attention zone. Schedule inspection within next maintenance window.",
            'WARNING':"Anomalous patterns or RUL below 60 cycles. Escalate to maintenance team.",
            'CRITICAL':"IMMEDIATE ACTION REQUIRED. Near-failure condition. Ground engine for inspection."}
        st.markdown(f'<div style="background:#111827;border:1px solid #1E2A3A;border-radius:12px;padding:16px 20px;margin:12px 0;"><div style="font-size:0.7rem;color:#8892A4;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;">Maintenance Recommendation</div><div style="color:#E2E8F0;font-size:0.9rem;line-height:1.6">{recs[status]}</div></div>', unsafe_allow_html=True)
        cg,cb = st.columns(2)
        with cg:
            fig_g = go.Figure(go.Indicator(mode="gauge+number", value=pred_rul,
                number=dict(suffix=" cy", font=dict(size=26, color='#E2E8F0')),
                title=dict(text="Predicted RUL", font=dict(size=13, color='#8892A4')),
                gauge=dict(axis=dict(range=[0,125], tickfont=dict(color='#8892A4', size=10)),
                    bar=dict(color=color, thickness=0.25), bgcolor='#0D1117', bordercolor='#1E2A3A',
                    steps=[dict(range=[0,15],color='rgba(239,68,68,0.15)'),dict(range=[15,30],color='rgba(249,115,22,0.12)'),dict(range=[30,60],color='rgba(245,158,11,0.1)'),dict(range=[60,125],color='rgba(16,185,129,0.08)')],
                    threshold=dict(line=dict(color='#EF4444', width=3), thickness=0.75, value=30))))
            lg = base_layout(); lg['height']=260
            fig_g.update_layout(**lg)
            st.plotly_chart(fig_g, use_container_width=True)
        with cb:
            sn = list(vals.keys()); sv = list(vals.values())
            fig_b = go.Figure(go.Bar(y=sn, x=sv, orientation='h', marker=dict(color=sv, colorscale=[[0,'#1A56DB'],[1,'#34D399']], showscale=False)))
            lb = base_layout(); lb['height']=260; lb['title']='Input Sensor Values'; lb['title_font_color']='#E2E8F0'
            fig_b.update_layout(**lb)
            fig_b.update_xaxes(gridcolor='#1E2A3A'); fig_b.update_yaxes(gridcolor='#1E2A3A')
            st.plotly_chart(fig_b, use_container_width=True)

with tab3:
    st.markdown('<div class="section-header">Model Performance Dashboard</div>', unsafe_allow_html=True)
    m = config['metrics']
    kpis = [("Detection Lead Time",f"{m['anomaly_lead_time']} cy","Target ≥ 15","#1A56DB",True),
        ("False Positive Rate",f"{m['anomaly_fpr']}%","Target < 5%","#10B981",True),
        ("Danger Zone Coverage",f"{m['anomaly_coverage']}%","Target ≥ 90%","#F59E0B",False),
        ("RUL MAE",f"{m['rul_mae']} cy","Target ≤ 20","#8B5CF6",True),
        ("Near-Failure MAE",f"{m['rul_mae_critical']} cy","Critical zone","#EF4444",True)]
    cols = st.columns(5)
    for col,(label,val,target,color,passed) in zip(cols,kpis):
        badge = "PASS" if passed else "PARTIAL"; bc = "#10B981" if passed else "#F59E0B"
        with col:
            st.markdown(f'<div class="kpi-card"><div style="font-size:0.64rem;color:#8892A4;text-transform:uppercase;letter-spacing:.05em">{label}</div><div class="kpi-value" style="color:{color};font-size:1.5rem">{val}</div><div style="font-size:0.7rem;color:#8892A4">{target}</div><div style="margin-top:6px;display:inline-block;border:1px solid {bc};border-radius:10px;padding:1px 10px;font-size:0.68rem;color:{bc};font-weight:600">{badge}</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Model Version Evolution</div>', unsafe_allow_html=True)
        versions=['V1','V2','V3']; cov=[28.3,60.3,m['anomaly_coverage']]; fpr=[0.73,4.81,m['anomaly_fpr']]
        fig_v = make_subplots(specs=[[{"secondary_y":True}]])
        fig_v.add_trace(go.Bar(name='Coverage (%)', x=versions, y=cov, marker=dict(color=['#374151','#1E3A5F','#1A56DB']), text=[f"{v}%" for v in cov], textposition='outside', textfont=dict(color='#E2E8F0')), secondary_y=False)
        fig_v.add_trace(go.Scatter(name='FPR (%)', x=versions, y=fpr, mode='lines+markers', line=dict(color='#34D399', width=2.5), marker=dict(size=9, color='#34D399')), secondary_y=True)
        fig_v.add_hline(y=90, line=dict(color='#10B981', dash='dot', width=1.5), secondary_y=False)
        lv = base_layout(); lv['height']=320; lv['showlegend']=True; lv['legend']=dict(orientation='h', y=1.12, font=dict(color='#E2E8F0'))
        fig_v.update_layout(**lv)
        fig_v.update_yaxes(title_text="Coverage (%)", color='#8892A4', gridcolor='#1E2A3A', secondary_y=False)
        fig_v.update_yaxes(title_text="FPR (%)", color='#8892A4', secondary_y=True)
        fig_v.update_xaxes(gridcolor='#1E2A3A')
        st.plotly_chart(fig_v, use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Acceptance Criteria</div>', unsafe_allow_html=True)
        for cid,name,res,ok in [("R1a","Anomaly Lead Time ≥ 15",f"{m['anomaly_lead_time']} cy",True),
            ("R1b","RUL MAE ≤ 20",f"{m['rul_mae']} cy",True),("R1c","Near-Failure MAE ≤ 20",f"{m['rul_mae_critical']} cy",True),
            ("R2","False Positive Rate < 5%",f"{m['anomaly_fpr']}%",True),("R3","Danger Zone Coverage ≥ 90%",f"{m['anomaly_coverage']}%",False),
            ("R4","Anomaly Rate ≤ 15%","14.0%",True)]:
            color = "#10B981" if ok else "#F59E0B"; icon = "✓" if ok else "~"
            st.markdown(f'<div class="criteria-row"><div><b style="color:{color}">[{icon}] {cid}</b> <span style="color:#E2E8F0;font-size:0.82rem;margin-left:6px">{name}</span></div><span style="color:{color};font-size:0.82rem;font-weight:600">{res}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Feature Importance — Top 10</div>', unsafe_allow_html=True)
    tf = [('sensor_4_roll10',0.3521),('sensor_4_roll5',0.2291),('sensor_11_roll10',0.0830),('sensor_17_roll10',0.0619),('sensor_11_roll5',0.0529),('sensor_3_roll10',0.0320),('sensor_2_roll10',0.0237),('sensor_17_roll5',0.0205),('sensor_12_roll10',0.0164),('sensor_21_roll5',0.0160)]
    fn = [f.replace('sensor_','S').replace('_roll10',' R10').replace('_roll5',' R5') for f,_ in tf]; fv=[v for _,v in tf]
    fc = ['#1A56DB' if i<2 else '#2E75B6' if i<5 else '#4B8EC0' for i in range(len(fv))]
    fig_i = go.Figure(go.Bar(x=fv[::-1], y=fn[::-1], orientation='h', marker=dict(color=fc[::-1]), text=[f"{v:.3f}" for v in fv[::-1]], textposition='outside', textfont=dict(color='#E2E8F0', size=10)))
    li = base_layout(); li['height']=340
    fig_i.update_layout(**li)
    fig_i.update_xaxes(title_text='Importance', color='#8892A4', gridcolor='#1E2A3A'); fig_i.update_yaxes(gridcolor='#1E2A3A')
    st.plotly_chart(fig_i, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Project Journey — Raw Data to Dual-Model System</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    for col,(v,l,c) in zip(cols,[("20,631","Records","#1A56DB"),("100","Engines","#0EA5E9"),("21","Raw Sensors","#8B5CF6"),("11","Informative","#10B981"),("9","ML Features","#F59E0B"),("45","Engineered","#EF4444")]):
        with col:
            st.markdown(f'<div style="text-align:center;padding:16px 8px;background:#111827;border-radius:10px;border:1px solid #1E2A3A;"><div style="font-size:1.5rem;font-weight:800;color:{c}">{v}</div><div style="font-size:0.7rem;color:#8892A4;margin-top:3px">{l}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Model Selection — All Algorithms Tested</div>', unsafe_allow_html=True)
    mc=['K-Means','One-Class SVM','IsoForest V1','IsoForest V2','IsoForest V3 (FINAL)']; fp=[33.3,4.5,0.73,4.81,4.54]; ld=[43.7,92.5,34.0,69.4,71.7]
    cc=['#EF4444','#F59E0B','#60A5FA','#818CF8','#34D399']; sl=[0,0,0,0,1]
    fig_c = go.Figure()
    for mm,f,l,c,s in zip(mc,fp,ld,cc,sl):
        fig_c.add_trace(go.Scatter(x=[f], y=[l], mode='markers+text', name=mm, text=[mm], textposition='top center', textfont=dict(size=9, color=c), marker=dict(size=20 if s else 13, color=c, symbol='star' if s else 'circle', line=dict(color='#0A0E1A', width=2))))
    fig_c.add_vrect(x0=0, x1=5, fillcolor='rgba(16,185,129,0.05)', line_width=0)
    lc = base_layout(); lc['height']=380; lc['showlegend']=False; lc['title']='FPR vs Detection Lead Time (top-left zone = best)'; lc['title_font_color']='#E2E8F0'
    fig_c.update_layout(**lc)
    fig_c.update_xaxes(title_text='False Positive Rate (%)', color='#8892A4', gridcolor='#1E2A3A')
    fig_c.update_yaxes(title_text='Detection Lead Time (cycles)', color='#8892A4', gridcolor='#1E2A3A')
    st.plotly_chart(fig_c, use_container_width=True)
    st.markdown('<div class="section-header">Key Achievements</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col,(v,d,c) in zip(cols,[("71.7 cy","Detection lead time — 4.8× above the 15-cycle requirement","#1A56DB"),("8.08 cy","Near-failure MAE — most accurate exactly when it matters","#10B981"),("4.54%","False positive rate — reliable alerts, no alarm fatigue","#8B5CF6"),("10 models","Total experiments across 3 algorithm families","#F59E0B")]):
        with col:
            st.markdown(f'<div style="background:#0D1117;border:1px solid #1E2A3A;border-radius:14px;padding:18px;text-align:center;height:150px;display:flex;flex-direction:column;justify-content:center;border-top:3px solid {c};"><div style="font-size:1.6rem;font-weight:800;color:{c}">{v}</div><div style="font-size:0.74rem;color:#8892A4;margin-top:6px;line-height:1.5">{d}</div></div>', unsafe_allow_html=True)

    # ── Credits & Contributions ──
    st.markdown('<div class="section-header">Project Contributions</div>', unsafe_allow_html=True)
    st.markdown('<p class="helptext">This system was developed by Group 2 for the Basic Design course at Sejong University. The complete project — from data analysis to the dual-model machine learning pipeline — was a team effort.</p>', unsafe_allow_html=True)

    team = [
        ("Arafat Mohammed","Project Lead & Lead ML Engineer","Led the ML pipeline, integrated the dual-model system, and designed & built this web application"),
        ("Latipov Javokhir","Lead ML Engineer","Co-developed model code, feature engineering, and RUL regression"),
        ("Bhuiyan Ahasanul Monir","Systems Architect","System architecture and functional decomposition design"),
        ("Esha Anika Tajnima","QA & Testing Engineer","Validation criteria, design targets, and acceptance testing"),
        ("Yusupjonov Otabek","Decision & Risk Analyst","Concept evaluation (AHP/ANP) and risk analysis"),
        ("Rubayed","Data Visualisation Specialist","Charts, visual comparisons, and benchmarking"),
        ("Arfin Ifthekhar","Technical Writer","Documentation and reporting"),
    ]
    rows = [team[i:i+2] for i in range(0, len(team), 2)]
    for row in rows:
        cols = st.columns(2)
        for col,(name,role,contrib) in zip(cols,row):
            with col:
                st.markdown(f"""
                <div style="background:#0D1117;border:1px solid #1E2A3A;border-radius:12px;
                            padding:14px 18px;margin:6px 0;border-left:3px solid #1A56DB;">
                    <div style="color:#E2E8F0;font-weight:700;font-size:0.92rem">{name}</div>
                    <div style="color:#60A5FA;font-size:0.76rem;margin:2px 0 6px">{role}</div>
                    <div style="color:#8892A4;font-size:0.78rem;line-height:1.5">{contrib}</div>
                </div>""", unsafe_allow_html=True)

    # ── Web app credit ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0D1B35,#0A0E1A);border:1px solid #1E2A3A;
                border-radius:16px;padding:24px 28px;margin-top:20px;text-align:center;
                position:relative;overflow:hidden;">
        <div style="position:absolute;top:-30px;right:-30px;width:160px;height:160px;
                    background:radial-gradient(circle,rgba(26,86,219,0.12),transparent 70%);border-radius:50%;"></div>
        <div style="position:relative;z-index:1;">
            <div style="font-size:0.72rem;color:#8892A4;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Web Application</div>
            <div style="font-size:1.1rem;color:#E2E8F0;font-weight:700;">
                Designed & Developed by <span style="background:linear-gradient(135deg,#1A56DB,#0EA5E9);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Arafat Mohammed</span>
            </div>
            <div style="font-size:0.82rem;color:#8892A4;margin-top:6px;">
                Project Lead · Group 2 · Sejong University · Basic Design Course
            </div>
            <div style="font-size:0.78rem;color:#8892A4;margin-top:10px;">
                Built with Streamlit · Powered by Isolation Forest + Gradient Boosting on NASA C-MAPSS data
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
