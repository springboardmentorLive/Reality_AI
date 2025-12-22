import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

def plot_segmentation(image, mask, classes):
    """
    Plots the original image and the segmentation mask.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')
    
    # Create a custom colormap or just show the index map
    im = ax[1].imshow(mask, cmap='viridis', vmin=0, vmax=len(classes)-1)
    ax[1].set_title("Segmentation Mask")
    ax[1].axis('off')
    
    # Add a legend/colorbar setup manually if needed, or just let colorbar handle it
    cbar = plt.colorbar(im, ax=ax[1], ticks=range(len(classes)))
    cbar.ax.set_yticklabels(classes)
    
    return fig

def plot_price_trend(df):
    """
    Plots the price trend using Plotly Candlestick.
    """
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color= '#2ecc71', # Emerald Green
                decreasing_line_color= '#e74c3c'  # Alizarin Red
                )])

    fig.update_layout(
        title={
            'text': 'Real Estate Price Forecast',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        yaxis_title='Price (₹)',
        xaxis_title='Date',
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600
    )
    return fig
