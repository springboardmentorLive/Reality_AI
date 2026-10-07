"""
RealtyAI Streamlit App Configuration & Theme Settings
"""

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SAMPLE_IMG_DIR = os.path.join(DATA_DIR, "sample_images")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")

PAGE_TITLE = "RealtyAI | Smart Real Estate Insight Platform"
PAGE_ICON = "🏢"
LAYOUT = "wide"

# Modern glassmorphic theme styling CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main {
        background: radial-gradient(circle at 10% 20%, rgba(17, 24, 39, 0.98) 0%, rgba(11, 15, 25, 1) 90.2%);
        color: #F3F4F6;
    }

    /* Glassmorphic card */
    .metric-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .metric-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .metric-delta-pos {
        color: #10B981;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }

    .metric-delta-neg {
        color: #EF4444;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #CBD5E1;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
    }

    /* Badge */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-blue { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-purple { background: rgba(168, 85, 247, 0.2); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }

    /* Custom Streamlit adjustments */
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
"""
