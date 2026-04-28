import io
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np, pandas as pd, streamlit as st

REQ = {"Date","Retail_Department","Service_Type","Region","Cost_INR","Usage_Units"}
DEPTS = ("Grocery","Fashion","Electronics","Home","Beauty","Pharmacy","Toys","Sports","SupplyChain","Marketing")
SVCS  = ("Compute","Storage","Database","Networking","Analytics","AI/ML","Security","Monitoring","CDN","Messaging")
REGS  = ("APAC","EMEA","NA","LATAM","IN")
SVC_SCALE = dict(zip(SVCS,[60,90,40,55,35,20,25,30,50,45]))
SVC_PRICE = dict(zip(SVCS,[3.6,1.8,4.2,2.6,5.0,8.5,3.1,2.2,2.0,2.4]))
REG_MULT  = dict(zip(REGS,[1.00,1.06,1.10,1.03,0.95]))

# ── Helpers ───────────────────────────────────────────────────────────────────
fmt      = lambda x: "₹0" if pd.isna(x) else f"₹{float(x):,.0f}"
safe_div = lambda n,d: 0.0 if not d or pd.isna(d) or n is None or pd.isna(n) else float(n)/float(d)
uniq     = lambda s: sorted(s.dropna().unique().tolist())
resample = lambda df,f: df.set_index("Date")["Cost_INR"].resample(f).sum().fillna(0).sort_index()

def coerce(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df.get("Date"), errors="coerce")
    for c in ("Cost_INR","Usage_Units"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df

def zscore(s, w=14):
    s = s.astype(float); kw=dict(window=w, min_periods=max(3,w//3))
    return ((s - s.rolling(**kw).mean()) / s.rolling(**kw).std(ddof=0).replace(0,pd.NA)).fillna(0)

def projection(daily):
    if daily.empty: return 0.0
    last=daily.index.max(); start=last.replace(day=1)
    end=pd.Timestamp((start+pd.offsets.MonthBegin(1)).to_pydatetime())-pd.Timedelta(days=1)
    return (float(daily.loc[start:last].sum())/max(1,(last-start).days+1))*((end-start).days+1)

def variance_drivers(df, dim, n=8):
    m = df.groupby(["Month",dim],dropna=False)["Cost_INR"].sum().reset_index()
    if m["Month"].nunique()<2: return pd.DataFrame()
    lm,pm = m["Month"].max(),(m["Month"].max()-pd.offsets.MonthBegin(1)).normalize()
    lv,pv = m[m.Month==lm].set_index(dim)["Cost_INR"], m[m.Month==pm].set_index(dim)["Cost_INR"]
    keys  = sorted(set(lv.index)|set(pv.index))
    out   = pd.DataFrame({dim:keys,"Prev":[float(pv.get(k,0)) for k in keys],"Last":[float(lv.get(k,0)) for k in keys]})
    out["Delta"]=out.Last-out.Prev; out=out.sort_values("Delta",ascending=False)
    out["Impact%"]=out.Delta.abs()/max(out.Delta.abs().sum(),1e-9)*100
    return out.head(n).reset_index(drop=True)

def idle_candidates(df):
    g=(df.groupby(["Retail_Department","Service_Type"],dropna=False)
         .agg(Cost_INR=("Cost_INR","sum"),Usage_Units=("Usage_Units","sum")).reset_index())
    if g.empty: return g
    g["Unit_Cost"]=(g.Cost_INR/g.Usage_Units.replace(0,pd.NA)).fillna(0)
    return g[(g.Cost_INR>=g.Cost_INR.quantile(0.75))&(g.Usage_Units<=g.Usage_Units.median())].sort_values("Cost_INR",ascending=False).head(10).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_data(b):
    try: return coerce(pd.read_csv(io.BytesIO(b) if b else "cloud_billing_500_rows.csv"))
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False)
def gen_csv(n=10_000):
    rng=np.random.default_rng(42); start=datetime(2025,1,1)
    dates=[start+timedelta(days=int(x)) for x in rng.integers(0,365,n)]
    dept=rng.choice(DEPTS,n,p=[0.18,0.14,0.12,0.10,0.08,0.08,0.06,0.06,0.10,0.08])
    svc =rng.choice(SVCS, n,p=[0.22,0.18,0.12,0.10,0.10,0.06,0.06,0.06,0.05,0.05])
    reg =rng.choice(REGS, n,p=[0.22,0.20,0.25,0.08,0.25])
    base=rng.lognormal(3.2,0.6,n)
    usage=np.array([base[i]*SVC_SCALE[svc[i]]/50 for i in range(n)])
    price=np.clip([SVC_PRICE[svc[i]]*REG_MULT[reg[i]]*rng.normal(1,.1) for i in range(n)],0.5,None)
    cost=usage*price; anom=rng.choice(n,size=max(20,n//400),replace=False); cost[anom]*=rng.uniform(2.5,6,len(anom))
    return pd.DataFrame({"Date":pd.to_datetime(dates),"Retail_Department":dept,"Service_Type":svc,
        "Region":reg,"Usage_Units":np.round(np.clip(usage,0,None),2),"Cost_INR":np.round(np.clip(cost,0,None),2)}
    ).sort_values("Date").to_csv(index=False).encode()

# ── Agents ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class F: agent:str; severity:str; title:str; why:str; recommendation:str; evidence:object=None

sev_color = lambda s: {"high":"#ef4444","medium":"#f59e0b"}.get((s or "").lower(),"#22c55e")

def budget_agent(daily,budget):
    if budget<=0: return[F("Budget Guard","low","No budget","Set a budget to enable alerts.","Add budget in sidebar.")]
    proj=projection(daily); burn=safe_div(proj,budget)*100; savings=max(0,proj-budget)
    sev="high" if burn>=110 else "medium" if burn>=95 else "low"
    return[F("Budget Guard",sev,"Budget breach likely" if burn>=110 else "Budget at risk" if burn>=95 else "Budget on track",
             f"Projected {fmt(proj)} = {burn:.1f}% of budget.",f"Cut {fmt(savings)} to stay within budget.",
             pd.DataFrame({"Projected":[proj],"Budget":[budget],"Usage%":[burn],"Savings":[savings]}))]

def anomaly_agent(daily,thr):
    if daily.empty: return[]
    z=zscore(daily,14); anom=pd.DataFrame({"Date":daily.index,"Cost":daily.values,"Z":z.values})
    anom=anom[anom.Z.abs()>=thr].sort_values("Z",ascending=False)
    if anom.empty: return[F("Anomaly Investigator","low","No anomalies",f"Spend within ±{thr:.1f}σ.","Lower threshold for more sensitivity.")]
    return[F("Anomaly Investigator","high" if len(anom)>=5 else "medium",f"{len(anom)} anomaly day(s)",
             "Daily spend deviated from rolling baseline.","Drill by service/department on those dates.",anom.head(15).reset_index(drop=True))]

def driver_agent(df):
    m=resample(df,"MS")
    if len(m)<2: return[F("Spend Driver","low","Not enough history","Need ≥2 months.","Expand date range.")]
    d=float(m.iloc[-1]-m.iloc[-2]); pct=safe_div(d,m.iloc[-2])*100
    drv=variance_drivers(df,"Service_Type",8)
    return[F("Spend Driver","high" if abs(pct)>=25 else "medium" if abs(pct)>=10 else "low",
             f"MoM: {fmt(d)} ({pct:+.1f}%)","Spend changed MoM; table shows top contributors.",
             "Focus on top positive deltas.",drv if not drv.empty else None)]

def unit_agent(df):
    u=(df.groupby("Service_Type",dropna=False).agg(Cost_INR=("Cost_INR","sum"),Usage_Units=("Usage_Units","sum"))
        .assign(Unit_Cost=lambda x:x.Cost_INR/x.Usage_Units.replace(0,pd.NA)).fillna(0).sort_values("Unit_Cost",ascending=False))
    if u.empty: return[]
    top=u.head(5).reset_index()
    return[F("Unit Economics","medium",f"Priciest: {top['Service_Type'].iloc[0]}","Some services have high cost per unit.","Review reserved/committed pricing.",top)]

def optim_agent(df):
    idle=idle_candidates(df)
    if idle.empty: return[F("Optimization Scout","low","No candidates","No high-spend+low-usage combos.","Broaden filters.")]
    return[F("Optimization Scout","high" if len(idle)>=5 else "medium","Idle/low-utilization candidates",
             "High-spend, low-usage pairs.","Stop or scale down idle workloads.",idle)]

AGENTS={"Budget Guard":budget_agent,"Anomaly Investigator":anomaly_agent,
        "Spend Driver":driver_agent,"Unit Economics":unit_agent,"Optimization Scout":optim_agent}

# ── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar():
    st.sidebar.header("Data")
    up=st.sidebar.file_uploader("Upload billing CSV",type=["csv"])
    df=load_data(up.getvalue() if up else None)
    with st.sidebar.expander("Download sample"):
        n=st.number_input("Rows",1000,50000,10000,1000)
        st.download_button("Download CSV",gen_csv(int(n)),f"cloud_billing_{int(n)}_rows.csv","text/csv",use_container_width=True)
    if df.empty: st.info("Upload CSV with: "+", ".join(f"`{c}`" for c in sorted(REQ))); st.stop()
    if miss:=sorted(REQ-set(df.columns)): st.error("Missing: "+", ".join(f"`{c}`" for c in miss)); st.stop()

    st.sidebar.header("Filters")
    dept=st.sidebar.multiselect("Department",uniq(df.Retail_Department),default=uniq(df.Retail_Department))
    svc =st.sidebar.multiselect("Service",   uniq(df.Service_Type),     default=uniq(df.Service_Type))
    reg =st.sidebar.multiselect("Region",    uniq(df.Region),           default=uniq(df.Region))
    dmin,dmax=df.Date.min(),df.Date.max()
    dr=st.sidebar.date_input("Date range",(dmin.date(),dmax.date()),dmin.date(),dmax.date())
    s,e=(pd.Timestamp(dr[0]),pd.Timestamp(dr[1])+pd.Timedelta(days=1)-pd.Timedelta(seconds=1)) if isinstance(dr,tuple) and len(dr)==2 else (pd.Timestamp(dmin),pd.Timestamp(dmax))
    fdf=df[df.Retail_Department.isin(dept)&df.Service_Type.isin(svc)&df.Region.isin(reg)&df.Date.between(s,e)].copy()
    if fdf.empty: st.error("No data matches filters."); st.stop()

    st.sidebar.header("Agents")
    enabled=st.sidebar.multiselect("Enable agents",list(AGENTS),default=list(AGENTS))
    budget=float(st.sidebar.number_input("Agent budget (INR)",0,step=10000))
    z=float(st.sidebar.slider("Agent Z-threshold",2.0,5.0,3.0,0.5))
    return df,fdf,enabled,budget,z

# ── Tabs ───────────────────────────────────────────────────────────────────────
def tab_overview(df):
    tc,tu=float(df.Cost_INR.sum()),float(df.Usage_Units.sum())
    monthly=resample(df,"MS"); mom="—"
    if len(monthly)>=2:
        d=float(monthly.iloc[-1]-monthly.iloc[-2]); mom=f"{fmt(d)} ({safe_div(d,monthly.iloc[-2])*100:+.1f}%)"
    top=df.groupby("Retail_Department").Cost_INR.sum().idxmax() if df.Retail_Department.nunique() else "—"
    for col,lbl,val in zip(st.columns(5),["Total Cost","Total Units","Avg Unit Cost","Top Dept","MoM"],
                           [fmt(tc),f"{tu:,.0f}",f"₹{safe_div(tc,tu):,.2f}",top,mom]): col.metric(lbl,val)
    st.divider()
    c1,c2=st.columns(2)
    with c1: st.subheader("Monthly Spend"); st.line_chart(monthly)
    with c2: st.subheader("Cost by Service"); st.bar_chart(df.groupby("Service_Type").Cost_INR.sum().sort_values(ascending=False))
    st.divider()
    c3,c4=st.columns(2)
    with c3: st.subheader("Regional Cost"); st.bar_chart(df.groupby("Region").Cost_INR.sum().sort_values(ascending=False))
    with c4:
        st.subheader("Unit Cost by Service")
        u=(df.groupby("Service_Type").agg(C=("Cost_INR","sum"),U=("Usage_Units","sum")).assign(UC=lambda x:x.C/x.U.replace(0,pd.NA)).fillna(0).sort_values("UC",ascending=False).head(12))
        st.bar_chart(u["UC"])
    st.divider(); st.subheader("Top 10 Transactions"); st.dataframe(df.nlargest(10,"Cost_INR"),use_container_width=True)

def tab_agents(df,enabled,budget,z):
    st.subheader("Cloud Cost Intelligence Agents")
    daily=resample(df,"D"); findings=[]
    for name in enabled:
        fn=AGENTS.get(name)
        if fn: findings+=(fn(daily,budget) if name=="Budget Guard" else fn(daily,z) if name=="Anomaly Investigator" else fn(df))
    if not findings: st.info("No findings. Enable at least one agent."); return
    for f in sorted(findings,key=lambda x:{"high":0,"medium":1,"low":2}.get((x.severity or "").lower(),9)):
        with st.container(border=True):
            st.markdown(f"**{f.agent}**  \n<span style='padding:2px 8px;border-radius:999px;background:{sev_color(f.severity)};color:white;font-size:12px'>{f.severity.upper()}</span>  \n**{f.title}**",unsafe_allow_html=True)
            st.write(f"**Why**: {f.why}"); st.write(f"**Rec**: {f.recommendation}")
            if f.evidence is not None and not f.evidence.empty:
                with st.expander("Evidence"): st.dataframe(f.evidence,use_container_width=True)

def tab_insights(df):
    st.subheader("Automated Insights")
    daily=resample(df,"D"); left,right=st.columns(2)
    with left:
        st.markdown("**Budget**")
        monthly=resample(df,"MS"); rec=int(max(0,float(monthly.iloc[-2]))) if len(monthly)>=2 else 0
        mode=st.radio("Mode",["Auto","Manual"],horizontal=True,key="bm")
        budget=rec if mode=="Auto" and rec>0 else st.number_input("Budget (INR)",0,step=10000,key="bi")
        if mode=="Auto" and rec>0: st.caption(f"Using: **{fmt(budget)}**")
        fc=projection(daily)
        if budget<=0: st.info("Set a budget to see forecast.")
        else:
            burn=safe_div(fc,budget)*100; a,b,c=st.columns(3)
            a.metric("Budget",fmt(budget)); b.metric("Forecast",fmt(fc)); c.metric("Savings needed",fmt(max(0,fc-budget)))
            st.progress(min(1.0,burn/100),text=f"Forecast: {burn:.0f}%")
            (st.error if burn>=110 else st.warning if burn>=95 else st.success)("Over budget." if burn>=110 else "Near budget." if burn>=95 else "On track.")
    with right:
        st.markdown("**Anomaly Detection**")
        thr=st.slider("Z-threshold",2.0,5.0,3.0,0.5)
        z=zscore(daily,14); anom=pd.DataFrame({"Date":daily.index,"Cost":daily.values,"Z":z.values})
        anom=anom[anom.Z.abs()>=thr].sort_values("Z",ascending=False)
        st.metric("Anomaly days",f"{len(anom):,}"); st.dataframe(anom.tail(12).sort_values("Date",ascending=False),use_container_width=True)
    st.divider(); st.markdown("**MoM Drivers**")
    for col,dim in zip(st.columns(2),["Service_Type","Retail_Department"]):
        with col:
            drv=variance_drivers(df,dim,10)
            if drv.empty: st.info("Need ≥2 months.")
            else: st.dataframe(drv.assign(Prev=drv.Prev.map(fmt),Last=drv.Last.map(fmt),Delta=drv.Delta.map(fmt),**{"Impact%":drv["Impact%"].map(lambda v:f"{v:.1f}%")}),use_container_width=True)

def tab_optimization(df):
    st.subheader("Optimization Opportunities")
    st.markdown("**1) Idle / low-utilization candidates**")
    idle=idle_candidates(df)
    if idle.empty: st.info("No idle candidates found.")
    else: st.dataframe(idle.assign(Cost_INR=idle.Cost_INR.map(fmt),Unit_Cost=idle.Unit_Cost.map(lambda v:f"₹{v:,.2f}"),Usage_Units=idle.Usage_Units.map(lambda v:f"{v:,.0f}")),use_container_width=True)
    st.divider(); st.markdown("**2) High unit-cost outliers**")
    u=(df.groupby(["Retail_Department","Service_Type"],dropna=False).agg(Cost_INR=("Cost_INR","sum"),Usage_Units=("Usage_Units","sum"))
        .assign(Unit_Cost=lambda x:x.Cost_INR/x.Usage_Units.replace(0,pd.NA)).fillna(0).sort_values("Unit_Cost",ascending=False).head(15).reset_index())
    st.dataframe(u.assign(Cost_INR=u.Cost_INR.map(fmt),Unit_Cost=u.Unit_Cost.map(lambda v:f"₹{v:,.2f}"),Usage_Units=u.Usage_Units.map(lambda v:f"{v:,.0f}")),use_container_width=True)
    st.divider(); st.markdown("**3) Quick actions**")
    for i in ["Commit budgets per dept/service.","Enforce cost tags (center, app, owner).","Review top unit-cost services for reserved pricing.","Investigate anomaly days for misconfigs/spikes."]:
        st.write(f"- {i}")

def tab_data(df,fdf):
    with st.expander("🔍 Deep Exploration"):
        st.subheader("Preview"); st.dataframe(df.head(),use_container_width=True)
        st.subheader("Describe"); st.dataframe(df.describe(include="all"),use_container_width=True)
        buf=io.StringIO(); df.info(buf=buf); st.text(buf.getvalue())
    st.subheader("Filtered Data"); st.dataframe(fdf,use_container_width=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Retail Cloud Cost Intelligence",layout="wide")
st.title("☁️ Retail Cloud Cost Intelligence")
df,fdf,agents,ab,az=sidebar()
for tab,fn in zip(st.tabs(["Overview","Agents","Insights","Optimization","Data"]),
    [lambda:tab_overview(fdf),lambda:tab_agents(fdf,agents,ab,az),
     lambda:tab_insights(fdf),lambda:tab_optimization(fdf),lambda:tab_data(df,fdf)]):
    with tab: fn()
