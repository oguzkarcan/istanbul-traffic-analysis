# Istanbul Traffic Analysis

## 🚦 Project Overview

This project explores **whether traffic patterns in Istanbul are influenced more by weather conditions, time factors, district location, or seasonal events**. 

By analyzing **traffic and weather data**, we aim to understand what factors most significantly impact traffic congestion in one of the world's most congested cities.

### **Factors Being Analyzed**

* **Weather Conditions** – How do precipitation, temperature, and other weather variables affect traffic density and average speeds?
* **Temporal Patterns** – Do rush hours, weekdays vs. weekends, or specific time periods show distinctive traffic patterns?
* **District Location** – Are certain districts consistently more congested? Do they respond differently to other variables?
* **Seasonal Effects** – Are there weekly or monthly patterns in traffic behavior that can be identified?

To analyze this, we collected and combined **traffic data from the Istanbul Metropolitan Municipality (IBB)** with **weather data from WeatherAPI.com**, examining how these factors interrelate and influence traffic conditions.

By identifying these patterns, we aim to provide insights that can help improve traffic management and urban planning in Istanbul.

## 🚦 Research Questions and Objectives

* Determine which factors (weather, time, location) have the strongest influence on Istanbul's traffic patterns
* Identify statistically significant relationships between specific weather conditions and traffic metrics
* Discover natural groupings and patterns in traffic data that may not be immediately apparent
* Apply advanced data science techniques including hypothesis testing, clustering, and dimensionality reduction to extract meaningful insights from traffic data
* Develop data-driven recommendations for traffic management and urban planning based on the findings

## 🚦 Motivation

Istanbul faces significant traffic congestion issues, consistently ranking among the world's most congested cities. Understanding the factors that affect traffic patterns is crucial for:

* Improving traffic management strategies
* Guiding urban planning and infrastructure development
* Enhancing public transportation systems
* Reducing economic costs associated with congestion
* Improving quality of life for Istanbul residents

This project applies data science techniques to generate actionable insights that can contribute to addressing these challenges.

## 🚦 Datasets & Data Collection

### Traffic Data
* **Source**: Istanbul Metropolitan Municipality (IBB) Open Data Portal
* **Metrics**: Traffic density, average speed, vehicle count
* **Coverage**: Hourly data from various districts

### Weather Data
* **Source**: WeatherAPI.com 
* **Metrics**: Temperature, humidity, wind speed, precipitation, weather condition
* **Coverage**: Hourly weather data for Istanbul

Both datasets were collected through API calls, cleaned, and merged to create a comprehensive dataset for analysis.

## 🚦 Hypotheses and Testing

1.  **H₁**: There is a significant difference in traffic density between weekdays and weekends.
    *   **Test Used**: Independent t-test
    *   **Result**: p-value = 0.448 (Not significant at α=0.05)
    *   **Conclusion**: Failed to reject the null hypothesis. No statistically significant difference found.
2.  **H₂**: Weather conditions have a significant effect on average traffic speed.
    *   **Test Used**: One-way ANOVA
    *   **Result**: p-value = 0.313 (Not significant at α=0.05)
    *   **Conclusion**: Failed to reject the null hypothesis. Weather conditions do not significantly affect average speed.
3.  **H₃**: There is a significant correlation between precipitation and traffic density.
    *   **Test Used**: Pearson correlation
    *   **Result**: Correlation = 0.091, p-value = 0.015 (Significant at α=0.05)
    *   **Conclusion**: Rejected the null hypothesis. There is a weak but statistically significant positive correlation.
4.  **H₄**: Rush hours show significantly different congestion patterns compared to non-rush hours.
    *   **Test Used**: Independent t-test
    *   **Result**: p-value = 0.244 (Not significant at α=0.05)
    *   **Conclusion**: Failed to reject the null hypothesis. No statistically significant difference found.

## 🚦 Dataset Analysis Plan

1.  **Data Preparation**: Clean, merge, and enrich datasets (e.g., temporal features, weather categories). Apply necessary transformations, avoiding direct use of raw data.
2.  **Exploratory Data Analysis (EDA)**: Visualize patterns (time, weather impact) and identify correlations.
3.  **Advanced Analysis**: Apply PCA, clustering, and seasonal decomposition.
4.  **Hypothesis Testing**: Statistically validate key relationships identified during EDA.

## 🚦 Tools and Technologies

- **Python 3.x** (Primary programming language)
- **Data Manipulation**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn, folium
- **Statistical Analysis**: scipy, statsmodels
- **Machine Learning**: scikit-learn
- **Data Collection**: requests (API interactions)

## 🚦 Analysis and Key Findings

*   **EDA Results**: Revealed distinct daily/weekly traffic patterns (e.g., morning/evening rushes) and identified key correlations (e.g., precipitation positively correlated with density, r=0.091, p=0.015).
*   **Hypothesis Testing**: Confirmed the significance of precipitation's effect on density. Other initial hypotheses (weekday vs. weekend density, weather effect on speed, rush hour difference) were not statistically significant with the enriched dataset.
*   **Advanced Insights**: Cluster analysis identified two main traffic states (free-flow vs. congested). District-level variations and weekly seasonality were also observed.

## Acknowledgments

This project was made possible by the contributions of the Istanbul Metropolitan Municipality and WeatherAPI.com. Special thanks to the data scientists and engineers who helped collect, clean, and analyze the data.
