# Istanbul Traffic Analysis

## Content
- [Project Overview](#-project-overview)
- [Research Questions and Objectives](#-research-questions-and-objectives) 
- [Motivation](#-motivation)
- [Datasets & Data Collection](#-datasets--data-collection)
- [Hypothesis](#-hypothesis)
- [Dataset Analysis Plan](#-dataset-analysis-plan)
- [Tools and Technologies](#-tools-and-technologies)
- [Project Structure](#-project-structure)
- [Analysis of Findings](#-analysis-of-findings)
- [Hypothesis Tests Results](#-hypothesis-tests-results)
- [Installation and Setup](#-installation-and-setup)
- [Usage](#-usage)

---

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

---

## 🚦 Research Questions and Objectives

* Determine which factors (weather, time, location) have the strongest influence on Istanbul's traffic patterns
* Identify statistically significant relationships between specific weather conditions and traffic metrics
* Discover natural groupings and patterns in traffic data that may not be immediately apparent
* Apply advanced data science techniques including hypothesis testing, clustering, and dimensionality reduction to extract meaningful insights from traffic data
* Develop data-driven recommendations for traffic management and urban planning based on the findings

---

## 🚦 Motivation

Istanbul faces significant traffic congestion issues, consistently ranking among the world's most congested cities. Understanding the factors that affect traffic patterns is crucial for:

* Improving traffic management strategies
* Guiding urban planning and infrastructure development
* Enhancing public transportation systems
* Reducing economic costs associated with congestion
* Improving quality of life for Istanbul residents

This project applies data science techniques to generate actionable insights that can contribute to addressing these challenges.

---

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

---

## 🚦 Hypothesis

1. **H₁**: There is a significant difference in traffic density between weekdays and weekends.
2. **H₂**: Weather conditions have a significant effect on average traffic speed.
3. **H₃**: There is a significant correlation between precipitation and traffic density.
4. **H₄**: Rush hours show significantly different congestion patterns compared to non-rush hours.

---

## 🚦 Dataset Analysis Plan

1. **Data Preprocessing**:
   * Clean and merge traffic and weather datasets
   * Handle missing values and outliers
   * Create temporal features (hour of day, day of week, etc.)

2. **Exploratory Data Analysis**:
   * Visualize traffic patterns by time of day and day of week
   * Examine relationships between weather variables and traffic metrics
   * Create correlation heatmaps to identify important relationships

3. **Advanced Analysis**:
   * Apply Principal Component Analysis (PCA) to identify key patterns
   * Conduct cluster analysis to discover natural traffic pattern groups
   * Perform seasonal decomposition to understand weekly patterns
   * Generate district-based heatmaps to understand geographical patterns

4. **Hypothesis Testing**:
   * Conduct statistical tests on key relationships
   * Validate or reject initial hypotheses
   * Document significance levels and effect sizes

---

## 🚦 Tools and Technologies

- **Python 3.x** (Primary programming language)
- **Data Manipulation**: pandas, numpy
- **Data Visualization**: matplotlib, seaborn, folium
- **Statistical Analysis**: scipy, statsmodels
- **Machine Learning**: scikit-learn
- **Data Collection**: requests (API interactions)

---

## 🚦 Project Structure
```
istanbul-traffic-analysis/
├── data/
│   ├── raw/                     # Raw data from APIs
│   └── processed/               # Cleaned and processed datasets
├── output/
│   ├── plots/                   # Visualizations and plots
│   └── results/                 # Analysis results and hypothesis tests
├── Istanbul_Traffic_Analysis.ipynb  # Main Jupyter Notebook for analysis
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

---

## 🚦 Analysis of Findings

### Traffic Patterns by Time

Our analysis revealed distinct traffic patterns throughout the day:
- **Morning Rush (7-10 AM)**: Shows high traffic density, particularly on weekdays
- **Evening Rush (4-7 PM)**: Demonstrates the highest congestion levels of the day
- **Weekend Patterns**: Generally lower traffic density but with different peak times

### Weather Impact on Traffic

Analysis of weather conditions showed:
- **Precipitation**: A statistically significant positive correlation with traffic density (r=0.091, p=0.015)
- **Temperature**: No significant direct effect on average speeds
- **Weather Conditions**: Different conditions show varying traffic patterns, though not statistically significant

### District-Based Analysis

Different districts show distinct congestion patterns:
- Some districts consistently show higher congestion during afternoon periods
- Others show more balanced traffic distribution throughout the day
- District-specific characteristics influence how traffic responds to other variables

### Cluster Analysis Results

We identified two distinct traffic pattern clusters:
- **Cluster 0**: Lower traffic density, higher average speeds (free-flowing traffic)
- **Cluster 1**: Higher traffic density, lower average speeds (congested conditions)

These clusters have different temporal distributions, with Cluster 1 appearing more frequently during rush hours and in high-density districts.

---

## 🚦 Hypothesis Tests Results

### Hypothesis 1: Weekday vs Weekend Traffic

- **Test Used**: Independent t-test
- **Result**: p-value = 0.448 (Not significant at α=0.05)
- **Conclusion**: Failed to reject the null hypothesis
- **Interpretation**: No statistically significant difference in traffic density between weekdays and weekends was found

### Hypothesis 2: Weather Conditions and Average Speed

- **Test Used**: One-way ANOVA
- **Result**: p-value = 0.313 (Not significant at α=0.05)
- **Conclusion**: Failed to reject the null hypothesis
- **Interpretation**: Weather conditions do not significantly affect average speed

### Hypothesis 3: Precipitation and Traffic Density

- **Test Used**: Pearson correlation
- **Result**: Correlation = 0.091, p-value = 0.015 (Significant at α=0.05)
- **Conclusion**: Rejected the null hypothesis
- **Interpretation**: There is a weak but statistically significant positive correlation between precipitation and traffic density

### Hypothesis 4: Rush Hours and Congestion

- **Test Used**: Independent t-test
- **Result**: p-value = 0.244 (Not significant at α=0.05)
- **Conclusion**: Failed to reject the null hypothesis
- **Interpretation**: No statistically significant difference in congestion during rush hours compared to non-rush hours

---

## 🚦 Installation and Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/istanbul-traffic-analysis.git
cd istanbul-traffic-analysis
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## 🚦 Usage
The main analysis is conducted within the Jupyter Notebook:

1.  **Launch Jupyter:**
    Ensure you have Jupyter Notebook or JupyterLab installed (`pip install notebook` or `pip install jupyterlab`).
    Navigate to the project directory in your terminal and run:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
2.  **Open and Run the Notebook:**
    Open the `Istanbul_Traffic_Analysis.ipynb` file in Jupyter.
    Run the cells sequentially to perform data loading, preprocessing, analysis, and visualization.

## Key Findings
- Identified a statistically significant positive correlation between precipitation and traffic density
- Discovered two distinct traffic pattern clusters with unique characteristics
- Found varying congestion patterns across different districts of Istanbul
- Detected weekly seasonality in traffic patterns

## Technologies Used
- Python 3.x
- pandas, numpy (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning)
- statsmodels (statistical analysis)
- scipy (statistical testing)
- folium (map visualizations)

## Data Sources
- Traffic data: Istanbul Metropolitan Municipality (IBB) Open Data Portal
- Weather data: WeatherAPI.com

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Istanbul Metropolitan Municipality for providing traffic data access
- WeatherAPI.com for weather data
- Project instructors and teaching assistants for guidance

## Contributors
- Your Name
