
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="TERA Environmental Command Centre",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top:1rem; max-width:1600px;}
.hero {
 background:linear-gradient(115deg,#10394a,#176b63);
 color:white;padding:22px;border-radius:15px;margin-bottom:15px;
}
.hero h1 {color:white;margin:0;font-size:2rem;}
.hero p {color:#d8f2ed;margin:6px 0 0 0;}
[data-testid="stMetric"] {
 background:white;border:1px solid #dce8e8;
 padding:12px;border-radius:12px;
}
[data-testid="stSidebar"] {background:#f2f8f7;}
</style>
<div class="hero">
<h1>🌿 TERA | Environmental Command Centre</h1>
<p>Technology for Environmental Risk Analysis • Greenovation Expo 2026</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "DEMONSTRATION MODE: This prototype uses illustrative zone values "
    "and synthetic training data. It is not a real disaster-warning system."
)

FACTORS = [
    "Hazard Severity", "Vulnerability", "Population Impact",
    "Ecological Value", "Urgency"
]
WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]

ZONES = {
    "Forest": ["Forest fire", 72, 65, 35, 95, 75],
    "Residential Area": ["Flooding", 80, 85, 95, 45, 90],
    "Hospital": ["Power disruption", 75, 90, 90, 30, 95],
    "Smart Farm": ["Water shortage", 60, 65, 45, 75, 60],
    "River & Dam": ["Rising water level", 85, 75, 70, 85, 85],
    "Industrial Zone": ["Chemical spill", 82, 70, 75, 65, 80]
}

def score(values):
    return float(np.dot(values, WEIGHTS))

def category(value):
    if value >= 80:
        return "Critical"
    if value >= 65:
        return "High"
    if value >= 45:
        return "Moderate"
    return "Lower"

# Train demonstration AI using synthetic examples.
@st.cache_resource
def train_demo_model():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 101, size=(3000, 5))
    y_scores = X @ np.array(WEIGHTS)
    y = np.array([category(s) for s in y_scores])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    return clf, accuracy

model, demo_accuracy = train_demo_model()

# Controls
st.sidebar.header("🎛️ Scenario Controls")
st.sidebar.caption("Adjust the selected zone and observe the changes.")
selected_zone = st.sidebar.selectbox("Choose a zone", list(ZONES.keys()))
base = ZONES[selected_zone]

values = []
for i, factor in enumerate(FACTORS):
    values.append(st.sidebar.slider(
        factor, 0, 100, int(base[i+1]),
        help=f"Weight: {WEIGHTS[i]*100:.0f}%"
    ))

st.sidebar.divider()
teams = st.sidebar.slider("Response teams available", 1, 6, 3)
vehicles = st.sidebar.slider("Vehicles available", 1, 6, 2)

# Score all zones; edited values apply to the selected zone.
records = []
for name, data in ZONES.items():
    vals = values if name == selected_zone else data[1:]
    s = score(vals)
    records.append({
        "Zone": name,
        "Hazard": data[0],
        "Priority Score": round(s, 1),
        "Priority Level": category(s),
        **dict(zip(FACTORS, vals))
    })

ranking = pd.DataFrame(records).sort_values(
    "Priority Score", ascending=False
).reset_index(drop=True)
ranking.insert(0, "Rank", range(1, len(ranking)+1))

top = ranking.iloc[0]
selected_score = score(values)
prediction = model.predict([values])[0]
probs = model.predict_proba([values])[0]
prob_df = pd.DataFrame({
    "Category": model.classes_,
    "Model output (%)": np.round(probs * 100, 1)
}).sort_values("Model output (%)", ascending=False)

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("🌍 Monitored zones", "6")
c2.metric("🚨 Highest priority", top["Zone"], f'{top["Priority Score"]}/100')
c3.metric("🤖 AI category", prediction, f"Model output: {max(probs)*100:.1f}%")
c4.metric("📊 Average score", f'{ranking["Priority Score"].mean():.1f}/100')

# Ranking and chart
left, right = st.columns([1.05, 1])
with left:
    st.subheader("🏆 Environmental Priority Ranking")
    st.dataframe(
        ranking[["Rank", "Zone", "Hazard", "Priority Score", "Priority Level"]],
        use_container_width=True, hide_index=True, height=280
    )

with right:
    st.subheader("📊 Priority Scores by Zone")
    chart = px.bar(
        ranking.sort_values("Priority Score"),
        x="Priority Score", y="Zone", color="Priority Level",
        orientation="h", text="Priority Score",
        color_discrete_map={
            "Critical": "#dc4c4c", "High": "#f28e2b",
            "Moderate": "#e7b938", "Lower": "#39a878"
        }
    )
    chart.update_layout(
        height=320, margin=dict(l=0,r=15,t=10,b=5),
        template="plotly_white", xaxis_title="Score out of 100",
        yaxis_title="", legend_orientation="h"
    )
    chart.update_traces(textposition="outside")
    st.plotly_chart(chart, use_container_width=True,
                    config={"displayModeBar": False})

# Explain selected score and AI
left, right = st.columns(2)
with left:
    st.subheader(f"💡 Why This Score? — {selected_zone}")
    contributions = pd.DataFrame({
        "Factor": FACTORS,
        "Input": values,
        "Weight": [f"{w*100:.0f}%" for w in WEIGHTS],
        "Contribution": [round(v*w, 2) for v, w in zip(values, WEIGHTS)]
    }).sort_values("Contribution", ascending=False)
    st.dataframe(contributions, use_container_width=True, hide_index=True)
    st.metric("Weighted priority score", f"{selected_score:.2f}/100",
              category(selected_score))
    st.progress(min(selected_score / 100, 1.0))

with right:
    st.subheader("🧠 AI Risk-Category Demonstration")
    st.write(f"Predicted category: **{prediction}**")
    st.caption(
        f"Test-set accuracy on synthetic examples: {demo_accuracy*100:.1f}%. "
        "This is not real-disaster prediction accuracy."
    )
    pfig = px.bar(
        prob_df.sort_values("Model output (%)"),
        x="Model output (%)", y="Category", orientation="h",
        color="Category",
        color_discrete_map={
            "Critical": "#dc4c4c", "High": "#f28e2b",
            "Moderate": "#e7b938", "Lower": "#39a878"
        }
    )
    pfig.update_layout(
        height=260, template="plotly_white",
        margin=dict(l=0,r=10,t=5,b=5), showlegend=False
    )
    st.plotly_chart(pfig, use_container_width=True,
                    config={"displayModeBar": False})

# Resource simulation
st.subheader("🚑 Limited Resource Allocation")
allocation = ranking[["Rank", "Zone", "Priority Score"]].copy()
allocation["Response Team"] = [
    f"Team {i+1}" if i < teams else "Not allocated"
    for i in range(6)
]
allocation["Vehicle"] = [
    f"Vehicle {i+1}" if i < vehicles else "Not allocated"
    for i in range(6)
]
st.dataframe(allocation, use_container_width=True, hide_index=True)

st.caption(
    "Allocation is an illustrative simulation, not real emergency-dispatch advice."
)

with st.expander("🗺️ Zone details and scoring formula"):
    st.write("The six zones represent a fictional demonstration city.")
    st.latex(r"S=0.30H+0.25V+0.20P+0.15E+0.10U")
    st.write("H = Hazard Severity; V = Vulnerability; P = Population Impact; "
             "E = Ecological Value; U = Urgency.")
    st.download_button(
        "⬇️ Download ranking CSV",
        ranking.to_csv(index=False).encode("utf-8"),
        "TERA_priority_ranking.csv", "text/csv"
    )

st.markdown("""
<div style="background:#10394a;color:#e6f5f3;padding:12px;
border-radius:10px;text-align:center;margin-top:12px">
TERA • Technology for Environmental Risk Analysis • Greenovation Expo 2026
</div>
""", unsafe_allow_html=True)
