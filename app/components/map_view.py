"""
Folium Interactive Map Component for RealtyAI
"""

import folium
from streamlit_folium import st_folium
import pandas as pd


METRO_COORDINATES = {
    "New York, NY": (40.7128, -74.0060),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Chicago, IL": (41.8781, -87.6298),
    "Dallas, TX": (32.7767, -96.7970),
    "Houston, TX": (29.7604, -95.3698),
    "Washington, DC": (38.9072, -77.0369),
    "Miami, FL": (25.7617, -80.1918),
    "Atlanta, GA": (33.7490, -84.3880),
    "San Francisco, CA": (37.7749, -122.4194),
    "Seattle, WA": (47.6062, -122.3321),
    "Austin, TX": (30.2672, -97.7431),
    "Boston, MA": (42.3601, -71.0589),
    "Denver, CO": (39.7392, -104.9903),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Charlotte, NC": (35.2271, -80.8431),
    "Baltimore, MD": (39.2904, -76.6122),
    "Cincinnati, OH": (39.1031, -84.5120),
    "Cleveland, OH": (41.4993, -81.6944),
}


def render_metro_market_map(summary_df, selected_metro=None, height=450):
    """
    Renders an interactive Folium dark matter map with regional housing valuations.
    """
    center_lat, center_lon = 39.50, -98.35
    zoom_start = 4

    if selected_metro and selected_metro in METRO_COORDINATES:
        lat, lon = METRO_COORDINATES[selected_metro]
        center_lat, center_lon = lat, lon
        zoom_start = 9

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB dark_matter",
        control_scale=True
    )

    for _, row in summary_df.iterrows():
        region = row["RegionName"]
        if region in METRO_COORDINATES:
            lat, lon = METRO_COORDINATES[region]
            val = row.get("LatestZHVI", 0)
            growth = row.get("YoY_Growth_Pct", 0)

            # Color coding based on price tier
            if val > 600000:
                color = "#EC4899"
            elif val > 400000:
                color = "#8B5CF6"
            elif val > 250000:
                color = "#3B82F6"
            else:
                color = "#10B981"

            popup_html = f"""
            <div style='font-family: Arial; min-width: 170px; color: #111;'>
                <h4 style='margin:0 0 6px 0; color: #1E293B;'>{region}</h4>
                <b>ZHVI Index:</b> ${val:,.0f}<br>
                <b>YoY Growth:</b> <span style='color: {"#10B981" if growth >= 0 else "#EF4444"}; font-weight:bold;'>{growth:+.1f}%</span><br>
                <b>State:</b> {row.get('StateName', 'US')}
            </div>
            """

            is_selected = (region == selected_metro)
            folium.CircleMarker(
                location=[lat, lon],
                radius=14 if is_selected else 8,
                color="#FFFFFF" if is_selected else color,
                weight=3 if is_selected else 1.5,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{region}: ${val:,.0f}"
            ).add_to(m)

    st_folium(m, width="100%", height=height, returned_objects=[])
