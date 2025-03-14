Istanbul Traffic & Weather Analysis
Overview
This project explores how weather conditions affect traffic density in Istanbul. We integrate traffic density data from the Istanbul Metropolitan Municipality Open Data Portal with weather data from Meteoroloji Genel Müdürlüğü to uncover trends and relationships.

Motivation
Istanbul faces significant congestion issues. By examining weather impacts (e.g., temperature, precipitation, wind speed) on traffic, the study aims to provide insights for urban planners and traffic management authorities to improve road safety and traffic flow.

Data Sources
Traffic Density Data
Sourced from the Istanbul Metropolitan Municipality Open Data Portal

Weather Data
Sourced from the Meteoroloji Genel Müdürlüğü Open Data Portal (URL subject to update)

Analysis & Methodology
Data Collection & Preprocessing
Both datasets were synchronized by time and location. Missing values were handled, and outliers removed.

Exploratory Data Analysis (EDA)
Visualizations (e.g., time series, heat maps) were used to identify trends and correlations between weather conditions and traffic density.

Statistical & Machine Learning Analysis
Correlation analysis and regression models were applied to quantify the impact of weather on traffic patterns.

Findings
Adverse weather conditions, such as heavy rainfall, correlate with reduced average speeds.
Seasonal variations significantly affect traffic density.
Weather variables can moderately predict changes in traffic conditions.
Limitations & Future Work
Data Resolution: Temporal and spatial granularity of the datasets may limit precision.
Model Enhancement: Future work could incorporate additional urban indicators such as event data or real-time traffic alerts.

Sources & Citations
Traffic Density Data:
https://data.ibb.gov.tr/dataset/hourly-traffic-density-data-set
Weather Data:
https://www.wunderground.com/history/daily/tr/istanbul
