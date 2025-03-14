# Istanbul Traffic & Weather Analysis Project

## Overview
This project explores how weather conditions affect traffic density in Istanbul. We integrate traffic density data from the Istanbul Metropolitan Municipality Open Data Portal with weather data from Meteoroloji Genel Müdürlüğü to uncover trends and relationships.

## Motivation
Istanbul faces significant congestion issues. By examining weather impacts (e.g., temperature, precipitation, wind speed) on traffic, the study aims to provide insights for urban planners and traffic management authorities to improve road safety and traffic flow.

## Data Sources
- **Traffic Density Data**  
  [Istanbul Metropolitan Municipality Open Data Portal](https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set)

- **Weather Data**  
  [Wunderground](https://www.wunderground.com/history/daily/tr/istanbul)

## Data Collection & Preprocessing
Both datasets were synchronized by matching timestamps and locations. Missing values were handled, and outliers removed to ensure consistency.

## Analysis & Methodology
1. **Exploratory Data Analysis (EDA)**  
   Initial visualizations (time series, heat maps) were used to identify trends and correlations between weather conditions and traffic density.

2. **Statistical & Machine Learning Analysis**  
   Correlation analysis and regression models were employed to quantify how weather factors influence traffic patterns.

## Findings
- **Adverse Weather:** Heavy rainfall and strong winds correlate with lower average speeds.
- **Seasonal Variations:** Summer and winter exhibit different peak congestion periods.
- **Predictive Potential:** Weather variables can moderately predict changes in traffic density.

## Limitations & Future Work
- **Data Resolution:** The temporal and spatial granularity of the datasets may limit precision.  
- **Model Enhancement:** Future plans include integrating real-time event data and other urban indicators for improved accuracy.
