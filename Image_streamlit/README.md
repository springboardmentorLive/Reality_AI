# Image & PDF Streamlit Utilities

This repository contains a collection of simple Streamlit applications for handling images and PDFs.

## Features

- **Image Gallery**: Display a collection of images in a grid layout.
- **Image Upload**: Upload and display images (JPG, PNG, JPEG).
- **PDF Upload**: Upload and view PDF files directly in the browser.
- **URL Image**: Display an image from a specifically provided URL.
- **Image Display**: Basic script to display a local image.

## Installation

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install streamlit pillow
```

## Usage

You can run each script individually using Streamlit:

### Image Gallery
Displays a static gallery of images (requires `sample.jpg` to be present).
```bash
streamlit run gallery.py
```

### Upload Image
Allows you to upload an image file and displays it.
```bash
streamlit run upload_image.py
```

### Upload PDF
Allows you to upload a PDF file and embeds it for viewing.
```bash
streamlit run upload_pdf.py
```

### Display Image
Displays a single specific image (`sample.jpg`).
```bash
streamlit run display_image.py
```

### URL Image
Displays an image fetched from a remote URL.
```bash
streamlit run url_image.py
```
