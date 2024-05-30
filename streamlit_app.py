import streamlit as st
import pandas as pd
import altair as alt
from statsmodels.tsa.arima.model import ARIMA

# Load your historical data or define your dataset
historical_data = pd.read_excel("Boston_Climate.xlsx")

# Convert data types to numeric if needed
historical_data = historical_data.apply(pd.to_numeric, errors='coerce')
historical_data.dropna(inplace=True)  # Drop rows with NaN values if any

# Streamlit app interface with CSS styling to remove space from the top
st.write('<style>body { margin: 0; padding: 0; }</style>', unsafe_allow_html=True)
st.write('<div style="text-align:center; color:black;"><h1><b>Average Temperature Forecasting App for Boston (from 2024)</b></h1></div>', unsafe_allow_html=True)

# Define the number of future periods (user input)
user_input = st.text_input('Enter future periods:', '10')
if user_input.isdigit():  # Check if the input is a valid integer
    num_periods = int(user_input)
else:
    st.error('Please enter a valid number of future periods.')
    st.stop()  # Stop script execution if input is invalid

# Fit the final model with the best parameters
final_model = ARIMA(historical_data['Temp_Avg'], order=(4, 1, 3))
final_fit_model = final_model.fit()

# Use the fitted model to forecast future periods
forecast_next_periods = final_fit_model.forecast(steps=num_periods)

# Generate future dates for the forecasted periods (first day of every month)
last_date = historical_data['Date'].max()
forecasted_dates = pd.date_range(start=last_date, periods=num_periods + 1, freq='MS')[1:]

# Combine forecasted dates with forecasted values
forecast_df = pd.DataFrame({'Date': forecasted_dates, 'Forecasted Temperature': forecast_next_periods})

# Reset index and set it starting from 1
forecast_df.reset_index(drop=True, inplace=True)
forecast_df.index = range(1, len(forecast_df) + 1)

# Format dates to display only date without timestamp
forecast_df['Date'] = forecast_df['Date'].dt.date

# Use Altair to create the line chart with custom axis labels
chart = alt.Chart(forecast_df).mark_line().encode(
    x=alt.X('Date:T', axis=alt.Axis(title='Month')),
    y=alt.Y('Forecasted Temperature:Q', axis=alt.Axis(title='Average Temperature')),
    tooltip=['Date:T', 'Forecasted Temperature:Q']
).properties(
    width=600,
    height=400
).configure_axis(
    labelFontSize=12,
    titleFontSize=14,
    grid=False,
    domain = True,
    domainWidth=0.8,
    domainColor='grey' 
)

# Use Streamlit columns to layout the components
col1, col2 = st.columns(2)

# Display the forecasted values and dates in a table format in the first column
with col1:
    st.write('**Forecasted Average Temperatures**', unsafe_allow_html=True)
    st.table(forecast_df)

# Line chart for forecasted values with date format on x-axis in the second column
with col2:
    st.write('**Forecasted Average Temperature Trends**', unsafe_allow_html=True)
    st.altair_chart(chart, use_container_width=True)
