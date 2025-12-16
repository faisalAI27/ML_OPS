import os
import requests
import streamlit as st
from datetime import datetime

BACKEND_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
CITIES = ["Islamabad", "Lahore", "Karachi", "Peshawar", "Quetta"]

st.set_page_config(
    page_title="SmogGuard PK",
    page_icon="🌫️",
    layout="wide",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #1f2937 0, #020617 45%, #000000 100%);
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617 0%, #030712 40%, #020617 100%);
}
.aq-card {
    border-radius: 18px;
    padding: 1.2rem 1.3rem;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    backdrop-filter: blur(8px);
}
.aq-label { text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.06em; color: #94a3b8; margin-bottom: 0.2rem; }
.aq-value-main { font-size: 2rem; font-weight: 700; margin: 0.2rem 0; }
.aq-sub { color: #cbd5e1; font-size: 0.9rem; }
.hazard-safe { border: 1px solid rgba(34,197,94,0.3); background: linear-gradient(135deg, rgba(34,197,94,0.08), rgba(22,163,74,0.12)); }
.hazard-bad { border: 1px solid rgba(239,68,68,0.35); background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(220,38,38,0.14)); }
.hazard-chip-safe { display:inline-block; padding:4px 10px; border-radius:12px; background:rgba(34,197,94,0.2); color:#bbf7d0; font-size:0.8rem; }
.hazard-chip-bad { display:inline-block; padding:4px 10px; border-radius:12px; background:rgba(239,68,68,0.2); color:#fecdd3; font-size:0.8rem; }
.aq-divider { height: 1px; margin: 1.5rem 0; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent); }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🌫️ SmogGuard PK")
st.caption("Real-time & 3-hour ahead AQI for major Pakistani cities (OpenWeather + ML model)")

with st.sidebar:
    st.header("City & Controls")
    selected_city = st.selectbox("Select city", CITIES, index=0)
    refresh = st.button("🔄 Refresh now")


def fmt_time(ts: str | None) -> str | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


@st.cache_data(ttl=120)
def fetch_city(city: str):
    url = f"{BACKEND_URL}/predict_realtime/{city}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json(), None
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = None
        msg = f"Backend error {resp.status_code}"
        if detail:
            msg = f"{msg}: {detail}"
        return None, msg
    except Exception as e:
        return None, str(e)


if refresh:
    fetch_city.clear()

data, error = fetch_city(selected_city)

if error or data is None:
    st.error(f"Could not load data for {selected_city}: {error}")
    st.stop()

pred = data.get("prediction", {}) or {}
rt = data.get("realtime", {}) or {}
meta = data.get("meta", {}) or {}
recs = data.get("recommendations", {}) or {}

aqi_3h = pred.get("aqi_3h")
hazard_prob = pred.get("hazard_prob")
hazard_label = pred.get("hazard_label")
current_aqi = rt.get("current_aqi")
observed_at_local = fmt_time(rt.get("observed_at_local"))
predicted_for_local = fmt_time(rt.get("predicted_for_local"))
scale = rt.get("scale", "OpenWeather AQI 1–5")
scale_desc = rt.get("scale_description", "")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="aq-card">
            <div class="aq-label">Current air quality – {selected_city}</div>
            <div class="aq-value-main">{current_aqi if current_aqi is not None else "N/A"}</div>
            <div class="aq-sub">{scale}{f" • {scale_desc}" if scale_desc else ""}</div>
            <div class="aq-sub">Observed at: {observed_at_local + " PKT" if observed_at_local else "N/A"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="aq-card">
            <div class="aq-label">Predicted AQI in 3 hours</div>
            <div class="aq-value-main">{f"{aqi_3h:.2f}" if isinstance(aqi_3h, (int, float)) else "N/A"}</div>
            <div class="aq-sub">Predicted for: {predicted_for_local + " PKT" if predicted_for_local else "N/A"}</div>
            <div class="aq-sub">Horizon: 3 hours ahead</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    is_hazard = str(hazard_label).lower() == "hazardous"
    hazard_class = "hazard-bad" if is_hazard else "hazard-safe"
    chip_class = "hazard-chip-bad" if is_hazard else "hazard-chip-safe"
    hazard_main = "⚠️ Hazardous" if is_hazard else "✔️ Safe / Moderate"
    hazard_prob_display = f"{hazard_prob*100:.1f}%" if isinstance(hazard_prob, (int, float)) else "N/A"
    st.markdown(
        f"""
        <div class="aq-card {hazard_class}">
            <div class="aq-label">Hazard status</div>
            <div class="aq-value-main">{hazard_main}</div>
            <div class="aq-sub"><span class="{chip_class}">Hazard probability: {hazard_prob_display}</span></div>
            <div class="aq-sub" style="margin-top:8px;">Values ≥ 4 on the 1–5 AQI scale are treated as 'Hazardous'.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="aq-divider"></div>', unsafe_allow_html=True)

# Recommendations section
if recs:
    sev = recs.get("severity_bucket", "Air quality guidance")
    headline = recs.get("headline")
    summary = recs.get("summary")
    actions_general = recs.get("actions_general", []) or []
    actions_sensitive = recs.get("actions_sensitive_groups", []) or []
    based_on = recs.get("based_on", {}) or {}
    horizon_hours = based_on.get("horizon_hours")

    badge_class = "hazard-bad" if str(sev).lower().startswith(("unhealthy", "very")) or "hazard" in str(sev).lower() else "hazard-safe"

    with st.container():
        st.subheader("What should you do now?")
        st.markdown(
            f"""
            <div class="aq-card {badge_class}">
              <div class="aq-label">{sev}</div>
              <div class="aq-value-main" style="font-size:1.2rem;">{headline or ""}</div>
              <div class="aq-sub">{summary or ""}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.markdown("**For everyone**")
            if actions_general:
                st.markdown("\n".join([f"- {item}" for item in actions_general]))
            else:
                st.markdown("_No specific actions available._")
        with col_rec2:
            st.markdown("**For sensitive groups**")
            if actions_sensitive:
                st.markdown("\n".join([f"- {item}" for item in actions_sensitive]))
            else:
                st.markdown("_No specific actions available._")

        if horizon_hours:
            st.caption(f"Based on forecast for the next {horizon_hours} hours.")

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("📊 Interpretation guide")
    st.markdown(
        """
        - 1 – Good
        - 2 – Fair
        - 3 – Moderate
        - 4 – Poor
        - 5 – Very Poor
        """
    )
    st.info(
        "Current AQI is from OpenWeather (observed now). The model predicts AQI 3 hours ahead using current pollutants + weather. Manual checks online usually show *current* AQI, not the model’s 3-hour forecast."
    )

with col_right:
    with st.expander(f"📄 Data details for {selected_city}", expanded=False):
        tab_human, tab_json = st.tabs(["Summary view", "Developer JSON"])

        def fmt_human_time(ts: str | None) -> str | None:
            if not ts:
                return None
            try:
                dt = datetime.fromisoformat(ts)
                return dt.strftime("%d %b %Y, %I:%M %p") + " PKT"
            except Exception:
                return ts

        def aqi_category(v):
            if v is None:
                return "Unknown"
            try:
                val = float(v)
            except Exception:
                return "Unknown"
            if val <= 1:
                return "Good"
            elif val <= 2:
                return "Fair"
            elif val <= 3:
                return "Moderate"
            elif val <= 4:
                return "Poor"
            else:
                return "Very Poor"

        current_cat = aqi_category(current_aqi)
        forecast_cat = aqi_category(aqi_3h)
        obs_time_h = fmt_human_time(rt.get("observed_at_local"))
        pred_time_h = fmt_human_time(rt.get("predicted_for_local"))
        model_version = meta.get("model_version", "unknown")

        with tab_human:
            st.markdown("**Quick summary**")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    f"""
                    <div class="aq-card">
                      <div class="aq-label">Current conditions</div>
                      <div class="aq-value-main">{current_aqi if current_aqi is not None else "N/A"}</div>
                      <div class="aq-sub">
                        Category: <b>{current_cat}</b><br/>
                        Observed at: {obs_time_h or "N/A"}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                hazard_prob_display = f"{hazard_prob*100:.1f}%" if isinstance(hazard_prob, (int, float)) else "N/A"
                st.markdown(
                    f"""
                    <div class="aq-card">
                      <div class="aq-label">3-hour forecast</div>
                      <div class="aq-value-main">{f"{aqi_3h:.2f}" if isinstance(aqi_3h, (int, float)) else "N/A"}</div>
                      <div class="aq-sub">
                        Expected category: <b>{forecast_cat}</b><br/>
                        Hazard status: {hazard_label or "N/A"}<br/>
                        Chance of hazardous air: {hazard_prob_display}<br/>
                        Predicted for: {pred_time_h or "N/A"}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(
                f"Model version: {model_version} • Threshold: AQI ≥ 4 on 1–5 scale is treated as 'Hazardous'."
            )

        with tab_json:
            st.json(data)
