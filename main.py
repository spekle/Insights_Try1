# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 16:44:45 2024

@author: User
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore

app = FastAPI()

# Pydantic model to handle user input
class InputData(BaseModel):
    months: List[str]  # List of months in 'YYYY-MM' format
    values: List[float]  # Corresponding values for those months

# Function to detect seasonality or outlier dynamically
def detect_seasonality_or_outlier(acf_values, data, threshold=0.2, min_lag=2, z_threshold=2.0):
    peaks = [i for i in range(min_lag, len(acf_values)) if acf_values[i] > threshold]
    if not peaks:
        return None, None

    peak_acf = max(peaks, key=lambda x: acf_values[x])

    # Group data by month and calculate mean
    monthly_means = data.groupby(data.index.month)['Value'].mean()

    # Calculate z-scores for monthly means to identify outliers
    monthly_zscores = zscore(monthly_means)

    # Check if any month is an outlier based on z-score threshold
    outlier_month = monthly_means.index[np.argmax(monthly_zscores)]
    if monthly_zscores.max() > z_threshold:
        return peak_acf, f"outlier in {pd.Timestamp(2024, outlier_month, 1).strftime('%B')}"

    seasonal_values = data['Value'][::peak_acf]
    if len(seasonal_values) < 2:
        return peak_acf, None

    seasonal_trend = "increase" if seasonal_values.diff().mean() > 0 else "decrease"
    return peak_acf, seasonal_trend

# Function to calculate 3-month moving average of slopes for a given year
def moving_average_slopes(data, year, window=3):
    data_year = data[data.index.year == year]
    slopes = np.diff(data_year['Value'])
    ma_slopes = np.convolve(slopes, np.ones(window) / window, mode='valid')
    
    if np.abs(ma_slopes.mean()) < 0.0001:
        ma_slope_trend = "The trend has stabilized since the beginning of the year"
    elif ma_slopes.mean() > 0:
        ma_slope_trend = "The trend has increased since the beginning of the year"
    else:
        ma_slope_trend = "The trend has decreased since the beginning of the year"
    
    return ma_slopes, ma_slope_trend

# Trend analysis and summary function
def trend_analysis(data, seasonal_period, seasonal_trend):
    x = np.arange(len(data))
    y = data['Value'].values
    slope = (y[-1] - y[0]) / (x[-1] - x[0])
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

    total_change = y[-1] - y[0]
    
    slopes = [data['Value'].iloc[i] - data['Value'].iloc[i-1] for i in range(1, len(data))]
    max_increase = max(slopes)
    max_decrease = min(slopes)
    max_increase_periods = [(data.index[i].strftime('%B %Y'), data.index[i+1].strftime('%B %Y')) for i, slope in enumerate(slopes) if slope == max_increase]
    max_decrease_periods = [(data.index[i].strftime('%B %Y'), data.index[i+1].strftime('%B %Y')) for i, slope in enumerate(slopes) if slope == max_decrease]

    max_increase_periods_str = ', '.join([f"{start} to {end}" for start, end in max_increase_periods])
    max_decrease_periods_str = ', '.join([f"{start} to {end}" for start, end in max_decrease_periods])

    ma_window = 3
    moving_avg = np.convolve(data['Value'].values, np.ones(ma_window) / ma_window, mode='valid')
    moving_avg_trend = (moving_avg[-1] - moving_avg[0]) / (len(moving_avg) - 1)
    ma_trend_direction = "upward" if moving_avg_trend > 0 else "downward" if moving_avg_trend < 0 else "stable"

    latest_year = data.index.year.max()
    ma_slopes_latest, ma_slope_trend_latest = moving_average_slopes(data, year=latest_year)

    if seasonal_period:
        if seasonal_trend.startswith('outlier'):
            seasonal_summary = f"An outlier has been detected in {seasonal_trend.split()[-1]} with a significant pattern repeating every {seasonal_period} months."
        else:
            seasonal_summary = f"A seasonal pattern repeating every {seasonal_period} months has been detected, with a general {seasonal_trend} in trend."
    else:
        seasonal_summary = "No seasonal pattern was detected, indicating an outlier."

    summary = (
        f"\n\n****SUMMARY****\nBy examining these insights, we can conclude that the overall value trend is "
        f"{direction} by {total_change:.4f} from start to end. The 3-month moving average indicates a {ma_trend_direction} trend, highlighting the general direction "
        f"of the data.\n\nSignificant fluctuations are evident, with the largest increase observed between "
        f"{max_increase_periods_str} (slope: {max_increase:.4f}) and the largest "
        f"decrease from {max_decrease_periods_str} (slope: {max_decrease:.4f})."
        f"\n\nAdditionally, {seasonal_summary}\n\n"
        f"**Trend for {latest_year}:** {ma_slope_trend_latest}."
    )

    return summary


# POST endpoint to receive user input




# Root endpoint (optional)
@app.get("/")
async def root():
    # Validate that the number of months matches the number of values
    if len(input_data.months) != len(input_data.values):
        raise HTTPException(status_code=400, detail="The number of months and values do not match.")

    try:
        # Convert input data into a Pandas DataFrame
        data = pd.DataFrame({
            'Month': pd.to_datetime(input_data.months),
            'Value': input_data.values
        })
        data.set_index('Month', inplace=True)

        # Perform trend analysis
        lag_acf = acf(data['Value'], nlags=24)
        seasonal_period, seasonal_trend = detect_seasonality_or_outlier(lag_acf, data)
        summary = trend_analysis(data, seasonal_period, seasonal_trend)

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
