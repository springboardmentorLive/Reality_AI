# Interactive Report Generator

A Streamlit application that allows users to generate and email PDF reports based on their input.

## Features

*   **Interactive Input**: Sidebar for entering personal details and comments.
*   **Real-time Preview**: View the report content as you type.
*   **PDF Generation**: Automatically creates a downloadable PDF report.
*   **Email Integration**: Send the generated report directly via Gmail using an App Password.

## Prerequisites

*   Python 3.x
*   A Google account (for email functionality) with 2-Step Verification enabled to generate an App Password.

## Installation

1.  Clone or download this repository.
2.  Install the required Python packages:

    ```bash
    pip install streamlit fpdf yagmail
    ```

## Usage

1.  Run the Streamlit app:

    ```bash
    streamlit run gmail.py
    ```

2.  Enter your details in the sidebar.
3.  Review the preview and click **Download PDF** to save the report.
4.  To email the report:
    *   Enter your Gmail address.
    *   Enter your **Google App Password** (NOT your regular password).
    *   Enter the recipient's email.
    *   Click **Send Email**.

> **Note:** To generate a Google App Password, go to your **Google Account > Security > 2-Step Verification > App Passwords**.
