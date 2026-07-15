"""
Streamlit Dashboard for Airbnb model serving.
Supports Price Prediction, Revenue Prediction, and Competitive Analysis.
"""

from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, Tuple

import plotly.graph_objects as go
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts" / "geoencoding"))
from geo_encoding import resolve_city_distance

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb Intelligence Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject Custom CSS ────────────────────────────────────────────────────────
st.markdown(_CSS := """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ─── Global Reset ─── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0c1222; }

/* hide default header & footer */
header[data-testid="stHeader"] { background: transparent !important; }
footer { display: none !important; }
#MainMenu { visibility: hidden; }

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1b33 0%, #0c1222 100%) !important;
    border-right: 1px solid rgba(148,163,184,.12);
}

/* ─── Glass Card ─── */
.glass {
    background: rgba(19,28,49,.55);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

/* ─── Hero Banner ─── */
.hero {
    background: linear-gradient(135deg, rgba(20,184,166,.18) 0%, rgba(56,189,248,.10) 50%, rgba(168,85,247,.10) 100%);
    border: 1px solid rgba(20,184,166,.20);
    border-radius: 20px;
    padding: 36px 40px 30px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(20,184,166,.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #5eead4, #38bdf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
    letter-spacing: -0.03em;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1.02rem;
    margin: 0;
    font-weight: 400;
}

/* ─── Status Badge ─── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: .82rem;
    font-weight: 600;
    letter-spacing: .01em;
}
.status-ok {
    background: rgba(16,185,129,.12);
    color: #34d399;
    border: 1px solid rgba(16,185,129,.25);
}
.status-err {
    background: rgba(239,68,68,.12);
    color: #f87171;
    border: 1px solid rgba(239,68,68,.25);
}

/* ─── Section Headers ─── */
.section-hdr {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 20px 0 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ─── Result Cards ─── */
.result-card {
    border-radius: 16px;
    padding: 22px 26px;
    margin: 12px 0;
    animation: fadeSlide .45s ease-out;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-ok {
    background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(20,184,166,.08));
    border: 1px solid rgba(16,185,129,.28);
    color: #a7f3d0;
}
.result-ok strong { color: #6ee7b7; font-size: 1.25rem; }
.result-ok .sub { color: #94a3b8; font-size: .88rem; margin-top: 4px; }
.result-err {
    background: linear-gradient(135deg, rgba(239,68,68,.12), rgba(244,63,94,.08));
    border: 1px solid rgba(239,68,68,.28);
    color: #fca5a5;
}
.result-err strong { color: #f87171; }

/* ─── Metric Mini Cards ─── */
.metric-row { display: flex; gap: 14px; margin: 14px 0; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 140px;
    background: rgba(19,28,49,.65);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 14px;
    padding: 16px 18px;
    text-align: center;
    transition: border-color .2s, transform .2s;
}
.metric-card:hover {
    border-color: rgba(20,184,166,.35);
    transform: translateY(-2px);
}
.metric-label { color: #64748b; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.metric-value { color: #e2e8f0; font-size: 1.25rem; font-weight: 700; margin-top: 4px; }

/* ─── Tabs ─── */
div[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: .92rem !important;
    padding: 10px 22px !important;
    border-radius: 10px !important;
    color: #94a3b8 !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    transition: all .25s !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #5eead4 !important;
    background: rgba(20,184,166,.10) !important;
    border-color: rgba(20,184,166,.30) !important;
}

/* ─── Form & Inputs ─── */
div[data-testid="stForm"] {
    background: rgba(19,28,49,.45);
    border: 1px solid rgba(148,163,184,.08);
    border-radius: 16px;
    padding: 24px;
}
.stNumberInput > div, .stTextInput > div, .stSelectbox > div {
    border-radius: 10px !important;
}

/* ─── Submit Button ─── */
div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    color: #fff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 0 !important;
    font-size: .95rem !important;
    letter-spacing: .02em;
    transition: opacity .2s, transform .15s !important;
}
div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
    opacity: .9;
    transform: scale(1.01);
}

/* ─── Expander ─── */
details[data-testid="stExpander"] {
    background: rgba(19,28,49,.35) !important;
    border: 1px solid rgba(148,163,184,.08) !important;
    border-radius: 12px !important;
}

/* ─── Divider ─── */
hr { border-color: rgba(148,163,184,.10) !important; }

/* ─── Footer ─── */
.dash-footer {
    text-align: center;
    color: #475569;
    font-size: .8rem;
    padding: 18px 0 8px;
}
</style>
""", unsafe_allow_html=True)

# ── API Config ────────────────────────────────────────────────────────────────
API_BASE_URL = st.secrets.get("api_url", "http://localhost:8000")
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/api/health",
    "price": f"{API_BASE_URL}/api/predict/price",
    "revenue": f"{API_BASE_URL}/api/predict/revenue",
    "competitive": f"{API_BASE_URL}/api/analyze/competitive",
}

CANCELLATION_LABELS = [
    "Full Refundable Until Check-in",
    "Full Refundable Until 24 Hours Before Check-in",
    "Full Refundable Until 72 Hours Before Check-in",
    "Refundable",
    "Flexible",
    "Moderate",
    "Limited",
    "Firm",
    "Strict",
    "Non-refundable",
    "Super Strict 30 Days",
    "Super Strict 60 Days",
]

PREDEFINED_AMENITIES = [
    "Wifi", "Kitchen", "Air conditioning", "Heating", "Refrigerator", "Essentials",
    "Portable fans", "Microwave", "Stove", "Oven", "Coffee maker", "Cooking basics",
    "Dishes and silverware", "Smoke alarm", "Fire extinguisher", "First aid kit",
    "Carbon monoxide alarm", "Patio or balcony", "Backyard", "Outdoor furniture", "Hammock",
    "Waterfront", "Lake access", "Crib", "High chair", "Children's dinnerware", "Baby bath",
    "Board games", "Long term stays allowed", "Luggage dropoff allowed", "Cleaning before checkout",
    "Outdoor kitchen", "Sauna", "Resort access", "Washer", "Dryer", "Free parking on premises",
    "Dedicated workspace", "Pool", "Hot tub", "Gym", "Beach access", "Pets allowed", "TV",
    "Elevator", "Balcony", "BBQ grill",
]

ROOM_TYPE_LABELS = {
    "Entire Home / Apt": "entire_home",
    "Hotel Room": "hotel_room",
    "Private Room": "private_room",
    "Shared Room": "shared_room",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def check_api_health() -> Tuple[bool, Dict]:
    try:
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": response.text}
    except Exception as exc:
        return False, {"error": str(exc)}


def call_api(endpoint: str, data: Dict) -> Dict:
    try:
        response = requests.post(endpoint, json=data, timeout=30)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        return {"success": False, "error": response.text}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Connection failed. Is FastAPI running on localhost:8000?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out. Please try again."}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def resolve_geo_preview(latitude: float, longitude: float) -> Dict:
    try:
        geo = resolve_city_distance(cityname=None, latitude=latitude, longitude=longitude)
        return {
            "nearest_city": str(geo.get("name") or "Unknown"),
            "zone": str(geo.get("zone") or "Unknown"),
            "distance_from_city_center": float(geo.get("distance_from_city_center") or geo.get("distance_km") or 0.0),
            "city_population": float(geo.get("city_population") or 0.0),
            "error": "",
        }
    except Exception as exc:
        return {
            "nearest_city": "Unknown",
            "zone": "Unknown",
            "distance_from_city_center": 0.0,
            "city_population": 0.0,
            "error": str(exc),
        }


def build_price_gauge(value: float, title: str) -> go.Figure:
    low = max(0.0, value * 0.70)
    high = value * 1.30
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value,
            number={"prefix": "$", "valueformat": ",.2f", "font": {"size": 38, "color": "#5eead4"}},
            title={"text": title, "font": {"size": 15, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [low, high], "tickcolor": "#334155", "tickwidth": 1},
                "bar": {"color": "#14b8a6", "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [low, value * 0.90], "color": "rgba(20,184,166,.08)"},
                    {"range": [value * 0.90, value * 1.10], "color": "rgba(20,184,166,.18)"},
                    {"range": [value * 1.10, high], "color": "rgba(56,189,248,.10)"},
                ],
                "threshold": {
                    "line": {"color": "#5eead4", "width": 3},
                    "thickness": 0.8,
                    "value": value,
                },
            },
        )
    )
    fig.update_layout(
        height=290,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif"},
    )
    return fig


def _metric_html(label: str, value: str) -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'</div>'
    )


def build_listing_inputs(prefix: str) -> Dict:
    """Render all listing input fields and return the payload dict."""

    st.markdown('<div class="section-hdr">🏡 Property Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        listing_type = st.text_input("Listing Type", value="House", key=f"{prefix}_listing_type")
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=50, value=2, step=1, key=f"{prefix}_bedrooms")
        photos_count = st.number_input("Photos Count", min_value=0, max_value=500, value=50, key=f"{prefix}_photos")
    with c2:
        room_type_label = st.selectbox("Room Type", list(ROOM_TYPE_LABELS.keys()), index=2, key=f"{prefix}_room_type")
        beds = st.number_input("Beds", min_value=0, max_value=50, value=3, step=1, key=f"{prefix}_beds")
        baths = st.number_input("Bathrooms", min_value=0, max_value=50, value=1, step=1, key=f"{prefix}_baths")
    with c3:
        guests = st.number_input("Max Guests", min_value=0, max_value=100, value=4, step=1, key=f"{prefix}_guests")
        min_nights = st.number_input("Min Nights", min_value=0, max_value=1000, value=2, key=f"{prefix}_min_nights")
        cancellation_policy = st.selectbox("Cancellation", CANCELLATION_LABELS, index=CANCELLATION_LABELS.index("Flexible"), key=f"{prefix}_cancel")

    room_type = ROOM_TYPE_LABELS[room_type_label]

    st.markdown('<div class="section-hdr">📍 Location & Fees</div>', unsafe_allow_html=True)
    c3a, c3b, c3c, c3d = st.columns(4)
    with c3a:
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=18.5204, format="%.6f", key=f"{prefix}_lat")
    with c3b:
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=73.8567, format="%.6f", key=f"{prefix}_lon")
    with c3c:
        cleaning_fee = st.number_input("Cleaning Fee ($)", min_value=0.0, max_value=10000.0, value=100.0, step=1.0, key=f"{prefix}_cleaning")
    with c3d:
        extra_guest_fee = st.number_input("Extra Guest Fee ($)", min_value=0.0, max_value=5000.0, value=20.0, step=1.0, key=f"{prefix}_extra_guest")

    geo = resolve_geo_preview(float(latitude), float(longitude))
    st.markdown(
        '<div class="metric-row">'
        + _metric_html("📌 Nearest City", geo["nearest_city"])
        + _metric_html("📏 Distance", f'{geo["distance_from_city_center"]:.2f} km')
        + _metric_html("👥 Population", f'{geo["city_population"]:,.0f}')
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-hdr">⭐ Reputation & Status</div>', unsafe_allow_html=True)
    c4, c5, c6, c7 = st.columns(4)
    with c4:
        avg_rating = st.number_input("Avg Rating", min_value=1.0, max_value=5.0, value=4.8, step=0.1, key=f"{prefix}_rating")
    with c5:
        num_reviews = st.number_input("Reviews", min_value=0, max_value=100000, value=100, key=f"{prefix}_reviews")
    with c6:
        superhost = st.selectbox("Superhost", ["No", "Yes"], index=1, key=f"{prefix}_superhost")
    with c7:
        registration = st.selectbox("Registered", ["No", "Yes"], index=1, key=f"{prefix}_registration")

    col_pm, _ = st.columns([1, 3])
    with col_pm:
        professional_management = st.selectbox("Pro Management", ["No", "Yes"], index=0, key=f"{prefix}_pro_mgmt")

    with st.expander("⚙️  Advanced Options"):
        ca, cb, cc = st.columns(3)
        with ca:
            city_name = st.text_input("City Name", value="Pune", key=f"{prefix}_city_name")
        with cb:
            ttm_blocked_days = st.number_input("TTM Blocked Days", min_value=0.0, value=0.0, step=1.0, key=f"{prefix}_ttm_blocked")
        with cc:
            ttm_total_days = st.number_input("TTM Total Days", min_value=1.0, value=365.0, step=1.0, key=f"{prefix}_ttm_total")

    st.markdown('<div class="section-hdr">🛋️ Amenities</div>', unsafe_allow_html=True)
    amenities = st.multiselect(
        "Select amenities",
        options=PREDEFINED_AMENITIES,
        default=["Wifi", "Kitchen", "Air conditioning", "Dedicated workspace"],
        placeholder="Start typing to filter…",
        key=f"{prefix}_amenities",
    )
    custom_amenities = st.text_input(
        "Custom amenities",
        value="",
        placeholder="e.g. Cable, Smart Lock",
        key=f"{prefix}_custom_amenities",
    )

    custom_list = [item.strip() for item in custom_amenities.split(",") if item.strip()]
    all_amenities = amenities + [item for item in custom_list if item not in amenities]

    payload = {
        "bedrooms": int(bedrooms),
        "beds": int(beds),
        "baths": int(baths),
        "guests": int(guests),
        "photos_count": int(photos_count),
        "superhost": 1 if superhost == "Yes" else 0,
        "num_reviews": int(num_reviews),
        "avg_rating": float(avg_rating),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "distance_from_city_center": float(geo["distance_from_city_center"]),
        "city_population": float(geo["city_population"]),
        "cancellation_policy": cancellation_policy,
        "min_nights": int(min_nights),
        "cleaning_fee": float(cleaning_fee),
        "extra_guest_fee": float(extra_guest_fee),
        "registration": 1 if registration == "Yes" else 0,
        "professional_management": 1 if professional_management == "Yes" else 0,
        "listing_type": listing_type,
        "room_type": room_type,
        "city_name": city_name,
        "amenities": all_amenities,
        "ttm_blocked_days": float(ttm_blocked_days),
        "ttm_total_days": float(ttm_total_days),
    }
    return payload


# ── Hero Banner ───────────────────────────────────────────────────────────────
is_healthy, health_data = check_api_health()

hero_badge = (
    '<span class="status-badge status-ok">● API Connected</span>'
    if is_healthy
    else '<span class="status-badge status-err">● API Offline</span>'
)

st.markdown(
    f"""
    <div class="hero">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
            <div>
                <div class="hero-title">Airbnb Intelligence Dashboard</div>
                <p class="hero-sub">Predict pricing, forecast revenue, and benchmark against the local market — all in one place.</p>
            </div>
            <div style="padding-top:6px;">{hero_badge}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not is_healthy:
    st.markdown(
        '<div class="result-card result-err"><strong>⚠ FastAPI backend is not reachable.</strong><br>'
        f'Start the API server and refresh this page.<br><code>{API_BASE_URL}</code></div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Tab Navigation ────────────────────────────────────────────────────────────
tab_price, tab_revenue, tab_competitive = st.tabs([
    "💰  Price Prediction",
    "📈  Revenue Prediction",
    "🏆  Competitive Analysis",
])

TAB_CONFIG = {
    "price": {"tab": tab_price, "endpoint": "price", "label": "Predict Price", "icon": "💰"},
    "revenue": {"tab": tab_revenue, "endpoint": "revenue", "label": "Predict Revenue", "icon": "📈"},
    "competitive": {"tab": tab_competitive, "endpoint": "competitive", "label": "Analyze Market", "icon": "🏆"},
}


def render_results(page_key: str, data: Dict):
    """Render beautiful result cards based on the page type."""
    if page_key in ("price", "revenue"):
        prediction = float(data.get("prediction", 0.0))
        formatted = data.get("prediction_formatted", f"${prediction:,.2f}")
        model_name = data.get("model", "Unknown")
        target = data.get("target", "N/A")

        st.markdown(
            f'<div class="result-card result-ok">'
            f'<strong>{formatted}</strong>'
            f'<div class="sub">Model: {model_name}  ·  Target: {target}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        chart_title = "Suggested Nightly Rate" if page_key == "price" else "Projected Annual Revenue"
        st.plotly_chart(build_price_gauge(prediction, chart_title), use_container_width=True)
    else:
        position = data.get("market_position", "N/A")
        score = data.get("competitiveness_score", 0)
        st.markdown(
            f'<div class="result-card result-ok">'
            f'<strong>Market Position: {position}</strong>'
            f'<div class="sub">Competitiveness Score: {score:.1f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="metric-row">'
            + _metric_html("Cluster", str(data.get("cluster_id", "-")))
            + _metric_html("Cluster Size", str(data.get("cluster_size", "-")))
            + _metric_html("Price vs Cluster", f'{data.get("price_vs_cluster", 0):.2f}x')
            + '</div>',
            unsafe_allow_html=True,
        )

    with st.expander("🔍 Raw API Response"):
        st.json(data)


for key, cfg in TAB_CONFIG.items():
    with cfg["tab"]:
        input_col, result_col = st.columns([1.3, 1.0], gap="large")

        with input_col:
            with st.form(f"{key}_form"):
                payload = build_listing_inputs(key)
                submit = st.form_submit_button(
                    f"{cfg['icon']}  {cfg['label']}",
                    use_container_width=True,
                )

        with result_col:
            st.markdown('<div class="section-hdr">📊 Results</div>', unsafe_allow_html=True)

            if submit:
                with st.spinner("Crunching numbers…"):
                    result = call_api(API_ENDPOINTS[cfg["endpoint"]], payload)

                if not result["success"]:
                    st.markdown(
                        f'<div class="result-card result-err">'
                        f'<strong>Request Failed</strong><br>{result.get("error", "Unknown error")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    render_results(key, result["data"])
            else:
                st.markdown(
                    '<div class="glass" style="text-align:center;padding:48px 24px;">'
                    '<div style="font-size:2.4rem;margin-bottom:10px;">🏠</div>'
                    '<div style="color:#94a3b8;font-size:.95rem;">Fill in listing details and submit to see predictions.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<div class="dash-footer">Airbnb Intelligence · {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
    unsafe_allow_html=True,
)
