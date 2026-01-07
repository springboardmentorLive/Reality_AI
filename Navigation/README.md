# Streamlit Navigation Patterns

This repository demonstrates three different navigation patterns for Streamlit applications.

## Prerequisites

Ensure you have the following installed:

```bash
pip install streamlit streamlit-option-menu
```

## files

1.  **Horizontal Bar (`horizontalbar.py`)**: Uses `streamlit-option-menu` to create a responsive horizontal navigation bar at the top of the page.
2.  **Sidebar (`sidebar.py`)**: Implements a classic sidebar navigation using `st.sidebar.selectbox`.
3.  **Top Bar (`topbar.py`)**: Shows a simple top-level radio button menu using native Streamlit widgets.

## Usage

To run any of the examples, use the `streamlit run` command:

**Horizontal Menu:**
```bash
streamlit run horizontalbar.py
```

**Sidebar Menu:**
```bash
streamlit run sidebar.py
```

**Top Radio Menu:**
```bash
streamlit run topbar.py
```
