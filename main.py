import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.stats import zscore
from typing import Optional
from fastapi import FastAPI

app = FastAPI()

# Load data - 2027 strike rate
data = pd.DataFrame({
    'Month': pd.to_datetime([
        '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06', '2021-07', '2021-08', '2021-09', '2021-10',
        '2021-11', '2021-12', '2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08',
        '2022-09', '2022-10', '2022-11', '2022-12', '2023-01', '2023-02', '2023-03', '2023-04', '2023-05', '2023-06',
        '2023-07', '2023-08', '2023-09', '2023-10', '2023-11', '2023-12', '2024-01', '2024-02', '2024-03', '2024-04',
        '2024-05', '2024-06', '2024-07', '2024-08']),
    'Value': [
        0.61417213, 0.604131535, 0.608632747, 0.590397396, 0.585851249, 0.592739726, 0.597486115, 0.601378518, 0.599316595,
        0.581071377, 0.60057554, 0.607168666, 0.61573472, 0.630873357, 0.624114299, 0.627539796, 0.622835151, 0.636592425, 
        0.441525896, 0.573272416, 0.630753667, 0.648900063, 0.633712308, 0.646197694, 0.65542055, 0.66864711, 0.665893271,
        0.665039188, 0.6624885, 0.656264501, 0.672882259, 0.669090909, 0.681241565, 0.677698017, 0.67190499, 0.661229458,
        0.662228451, 0.6395536, 0.643461796, 0.640921558, 0.643305826, 0.639287532, 0.669458953, 0.663475106
    ]
})
data.set_index('Month', inplace=True)

# Compute the autocorrelation function (ACF)
lag_acf = acf(data['Value'], nlags=24)

# Function to detect seasonality or outliers
def detect_seasonality_or_outlier(acf_values, data, threshold=0.2, min_lag=2, z_threshold=2.0):
    peaks = [i for i in range(min_lag, len(acf_values)) if acf_values[i] > threshold]
    if not peaks:
        return None, None

    peak_acf = max(peaks, key=lambda x: acf_values[x])
    monthly_means = data.groupby(data.index.month)['Value'].mean()
    monthly_zscores = zscore(monthly_means)
    outlier_month = monthly_means.index[np.argmax(monthly_zscores)]
    
    if monthly_zscores.max() > z_threshold:
        return peak_acf, f"outlier in {pd.Timestamp(2024, outlier_month, 1).strftime('%B')}"
    
    seasonal_values = data['Value'][::peak_acf]
    if len(seasonal_values) < 2:
        return peak_acf, None
    
    seasonal_trend = "increase" if seasonal_values.diff().mean() > 0 else "decrease"
    return peak_acf, seasonal_trend

# Detect seasonality or outlier
seasonal_period, seasonal_trend = detect_seasonality_or_outlier(lag_acf, data)

# Function to calculate moving average slopes
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

# Trend analysis function
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

# FastAPI endpoint to get the summary report
@app.get("/")
async def root():
    summary = trend_analysis(data, seasonal_period, seasonal_trend)
    return {"summary": summary}

