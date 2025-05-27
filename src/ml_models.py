"""
Machine Learning Models for Istanbul Traffic Analysis

This module implements:
- Clustering analysis (K-means, DBSCAN)
- Dimensionality reduction (PCA)
- Predictive models (Random Forest, SVM)
- Model evaluation and validation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, regression_report, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TrafficMLAnalyzer:
    """Machine Learning analysis for traffic data"""
    
    def __init__(self, data_path='data/processed/enriched_data.csv'):
        self.data = pd.read_csv(data_path)
        self.data['datetime'] = pd.to_datetime(self.data['datetime'])
        self.scaler = StandardScaler()
        self.pca = None
        self.models = {}
        self.results = {}
        
    def prepare_features(self):
        """Prepare features for ML analysis"""
        print("Preparing features for ML analysis...")
        
        # Select numerical features
        numerical_features = ['traffic_density', 'average_speed', 'vehicle_count',
                            'temperature', 'humidity', 'precipitation', 'wind_speed',
                            'hour', 'day_of_week', 'month', 'traffic_efficiency']
        
        # Select categorical features to encode
        categorical_features = ['district', 'weather_category', 'time_period', 
                              'temp_category', 'precipitation_category']
        
        # Filter available columns
        available_numerical = [col for col in numerical_features if col in self.data.columns]
        available_categorical = [col for col in categorical_features if col in self.data.columns]
        
        # Create feature matrix
        feature_data = self.data[available_numerical].copy()
        
        # Encode categorical variables
        label_encoders = {}
        for col in available_categorical:
            le = LabelEncoder()
            feature_data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
            label_encoders[col] = le
        
        # Handle missing values
        feature_data = feature_data.fillna(feature_data.median())
        
        print(f"Feature matrix shape: {feature_data.shape}")
        print(f"Features: {list(feature_data.columns)}")
        
        return feature_data, label_encoders
        
    def perform_clustering(self, n_clusters=3):
        """Perform K-means clustering analysis"""
        print(f"\nPerforming K-means clustering (k={n_clusters})...")
        
        features, _ = self.prepare_features()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        
        # Add cluster labels to data
        self.data['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_data = self.data[self.data['cluster'] == i]
            cluster_analysis[f'Cluster_{i}'] = {
                'size': len(cluster_data),
                'avg_density': cluster_data['traffic_density'].mean(),
                'avg_speed': cluster_data['average_speed'].mean(),
                'avg_temperature': cluster_data['temperature'].mean(),
                'common_weather': cluster_data['weather_category'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'common_district': cluster_data['district'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
            }
        
        result = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_analysis': cluster_analysis,
            'inertia': kmeans.inertia_
        }
        
        self.models['kmeans'] = kmeans
        self.results['clustering'] = result
        
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Inertia: {kmeans.inertia_:.2f}")
        
        return result
        
    def perform_dbscan_clustering(self, eps=0.5, min_samples=5):
        """Perform DBSCAN clustering"""
        print(f"\nPerforming DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        
        features, _ = self.prepare_features()
        scaled_features = self.scaler.fit_transform(features)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(scaled_features)
        
        # Count clusters (excluding noise points labeled as -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate silhouette score (only if we have clusters)
        silhouette_avg = silhouette_score(scaled_features, cluster_labels) if n_clusters > 1 else -1
        
        result = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette_avg,
            'eps': eps,
            'min_samples': min_samples
        }
        
        self.models['dbscan'] = dbscan
        self.results['dbscan'] = result
        
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        print(f"Silhouette Score: {silhouette_avg:.3f}" if silhouette_avg > -1 else "Silhouette Score: N/A")
        
        return result
        
    def perform_pca(self, n_components=5):
        """Perform Principal Component Analysis"""
        print(f"\nPerforming PCA (n_components={n_components})...")
        
        features, _ = self.prepare_features()
        scaled_features = self.scaler.fit_transform(features)
        
        # PCA
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(scaled_features)
        
        # Calculate explained variance
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Feature importance in each component
        feature_names = features.columns
        components_df = pd.DataFrame(
            self.pca.components_,
            columns=feature_names,
            index=[f'PC{i+1}' for i in range(n_components)]
        )
        
        result = {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'components': components_df,
            'transformed_features': pca_features
        }
        
        self.results['pca'] = result
        
        print(f"Explained variance by components: {explained_variance_ratio}")
        print(f"Cumulative explained variance: {cumulative_variance[-1]:.3f}")
        
        return result
        
    def train_traffic_density_predictor(self):
        """Train model to predict traffic density"""
        print("\nTraining traffic density prediction model...")
        
        features, _ = self.prepare_features()
        target = self.data['traffic_density']
        
        # Remove target from features if present
        if 'traffic_density' in features.columns:
            features = features.drop('traffic_density', axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Predictions and evaluation
        train_score = rf_model.score(X_train_scaled, y_train)
        test_score = rf_model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        result = {
            'model_type': 'Random Forest Regressor',
            'train_score': train_score,
            'test_score': test_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        self.models['density_predictor'] = rf_model
        self.results['density_prediction'] = result
        
        print(f"Train R²: {train_score:.3f}")
        print(f"Test R²: {test_score:.3f}")
        print(f"CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return result
        
    def train_weather_classifier(self):
        """Train model to classify weather conditions"""
        print("\nTraining weather classification model...")
        
        features, _ = self.prepare_features()
        
        # Use weather features to predict weather category
        weather_features = ['temperature', 'humidity', 'precipitation', 'wind_speed']
        available_weather_features = [col for col in weather_features if col in features.columns]
        
        X = features[available_weather_features]
        y = self.data['weather_category']
        
        # Remove any NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_train)
        
        # Predictions and evaluation
        train_score = rf_classifier.score(X_train_scaled, y_train)
        test_score = rf_classifier.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(rf_classifier, X_train_scaled, y_train, cv=5)
        
        result = {
            'model_type': 'Random Forest Classifier',
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classes': le.classes_
        }
        
        self.models['weather_classifier'] = rf_classifier
        self.results['weather_classification'] = result
        
        print(f"Train Accuracy: {train_score:.3f}")
        print(f"Test Accuracy: {test_score:.3f}")
        print(f"CV Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
        
        return result
        
    def optimize_clustering(self, max_k=10):
        """Find optimal number of clusters using elbow method"""
        print(f"\nOptimizing number of clusters (k=2 to {max_k})...")
        
        features, _ = self.prepare_features()
        scaled_features = self.scaler.fit_transform(features)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))
        
        # Find optimal k (highest silhouette score)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        
        result = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k,
            'best_silhouette': max(silhouette_scores)
        }
        
        self.results['clustering_optimization'] = result
        
        print(f"Optimal k: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.3f}")
        
        return result
        
    def run_all_ml_analyses(self):
        """Run all machine learning analyses"""
        print("=" * 60)
        print("MACHINE LEARNING ANALYSIS")
        print("=" * 60)
        
        # Clustering optimization
        self.optimize_clustering()
        
        # Clustering with optimal k
        if 'clustering_optimization' in self.results:
            optimal_k = self.results['clustering_optimization']['optimal_k']
            self.perform_clustering(n_clusters=optimal_k)
        else:
            self.perform_clustering(n_clusters=3)
        
        # DBSCAN clustering
        self.perform_dbscan_clustering()
        
        # PCA
        self.perform_pca()
        
        # Predictive models
        self.train_traffic_density_predictor()
        self.train_weather_classifier()
        
        # Summary
        self.print_ml_summary()
        
    def print_ml_summary(self):
        """Print summary of ML results"""
        print("\n" + "=" * 60)
        print("MACHINE LEARNING SUMMARY")
        print("=" * 60)
        
        if 'clustering' in self.results:
            cluster_result = self.results['clustering']
            print(f"\nK-Means Clustering:")
            print(f"  Optimal clusters: {cluster_result['n_clusters']}")
            print(f"  Silhouette score: {cluster_result['silhouette_score']:.3f}")
            
        if 'pca' in self.results:
            pca_result = self.results['pca']
            print(f"\nPCA Analysis:")
            print(f"  Components: {len(pca_result['explained_variance_ratio'])}")
            print(f"  Total variance explained: {pca_result['cumulative_variance'][-1]:.3f}")
            
        if 'density_prediction' in self.results:
            pred_result = self.results['density_prediction']
            print(f"\nTraffic Density Prediction:")
            print(f"  Model: {pred_result['model_type']}")
            print(f"  Test R²: {pred_result['test_score']:.3f}")
            
        if 'weather_classification' in self.results:
            class_result = self.results['weather_classification']
            print(f"\nWeather Classification:")
            print(f"  Model: {class_result['model_type']}")
            print(f"  Test Accuracy: {class_result['test_accuracy']:.3f}")

def main():
    """Main ML analysis pipeline"""
    print("Starting Machine Learning Analysis...\n")
    
    # Initialize analyzer
    analyzer = TrafficMLAnalyzer()
    
    # Run all analyses
    analyzer.run_all_ml_analyses()
    
    print("\nMachine Learning analysis completed!")

if __name__ == "__main__":
    main() 