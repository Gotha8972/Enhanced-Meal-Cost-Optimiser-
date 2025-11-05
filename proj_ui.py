# [file name]: proj_ui.py
"""
Enhanced Meal Economics Dashboard with ML Integration
Complete solution with realistic pricing, improved clustering, cost recommendations, and decision trees
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced Data Loader with Fallbacks
class EnhancedDataLoader:
    """Enhanced data loader with better food matching and fallback mechanisms."""
    
    def __init__(self):
        # Get the directory where this script is located
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.food_groups = self.load_food_groups()
        self.market_data = self.load_market_data()
        self.food_mappings = self.create_food_mappings()
    
    def create_food_mappings(self):
        """Create flexible mappings for food items."""
        return {
            'pea potato sandwich': 'sandwich',
            'bread and butter pudding': 'bread',
            'sambar': 'lentil_curry',
            'rice flakes': 'poha',
            'banana milkshake': 'banana',
            'black channa curry': 'chickpea_curry',
            'chicken curry': 'chicken_curry',
            'matar aloo': 'potato_pea',
            'chicken breast': 'chicken',
            'paneer curry': 'paneer'
        }
    
    def load_food_groups(self):
        """Load food groups from INDB.csv file with fallback to nutrition_data.csv and hardcoded data."""
        indb_csv_file = os.path.join(self.base_dir, 'INDB.csv')
        nutrition_file = os.path.join(self.base_dir, 'nutrition_data.csv')
        
        # Try loading from INDB.csv first
        try:
            if os.path.exists(indb_csv_file):
                df = pd.read_csv(indb_csv_file)
                
                if df is not None and not df.empty:
                    food_groups = {}
                    
                    # Map INDB.csv column names to our format
                    name_cols = ['food_name', 'food_name', 'food', 'item', 'name']
                    name_col = next((col for col in name_cols if col in df.columns), 'food_name')
                    
                    # INDB.csv uses these column names
                    protein_col = 'protein_g' if 'protein_g' in df.columns else None
                    carbs_col = 'carb_g' if 'carb_g' in df.columns else None
                    fats_col = 'fat_g' if 'fat_g' in df.columns else None
                    fiber_col = 'fibre_g' if 'fibre_g' in df.columns else ('fiber_g' if 'fiber_g' in df.columns else None)
                    calcium_col = 'calcium_mg' if 'calcium_mg' in df.columns else None
                    calories_col = 'energy_kcal' if 'energy_kcal' in df.columns else ('calories' if 'calories' in df.columns else None)
                    
                    for _, row in df.iterrows():
                        food_item = str(row[name_col]).lower().strip()
                        if pd.isna(food_item) or food_item == 'nan':
                            continue
                        
                        # Convert calcium from mg - keep as mg for consistency
                        calcium_val = float(row[calcium_col]) if calcium_col and pd.notna(row.get(calcium_col, 0)) else 0
                        # Keep calcium in mg (values are typically in mg range like 14.20, 20.87, etc.)
                        
                        food_groups[food_item] = {
                            'protein': float(row[protein_col]) if protein_col and pd.notna(row.get(protein_col, 0)) else 0,
                            'carbs': float(row[carbs_col]) if carbs_col and pd.notna(row.get(carbs_col, 0)) else 0,
                            'fats': float(row[fats_col]) if fats_col and pd.notna(row.get(fats_col, 0)) else 0,
                            'fiber': float(row[fiber_col]) if fiber_col and pd.notna(row.get(fiber_col, 0)) else 0,
                            'calcium': calcium_val,  # Keep in mg
                            'calories': float(row[calories_col]) if calories_col and pd.notna(row.get(calories_col, 0)) else 0
                        }
                    
                    if food_groups:
                        logger.info(f"Loaded {len(food_groups)} food items from INDB.csv")
                        return food_groups
        except Exception as e:
            logger.warning(f"Error loading INDB.csv: {e}, trying nutrition_data.csv fallback")
        
        # Fallback to nutrition_data.csv file
        try:
            if os.path.exists(nutrition_file):
                df = pd.read_csv(nutrition_file)
                food_groups = {}
                
                for _, row in df.iterrows():
                    food_item = str(row['food_item']).lower().strip()
                    food_groups[food_item] = {
                        'protein': float(row.get('protein', 0)),
                        'carbs': float(row.get('carbs', 0)),
                        'fats': float(row.get('fats', 0)),
                        'fiber': float(row.get('fiber', 0)),
                        'calcium': float(row.get('calcium', 0)),
                        'calories': float(row.get('calories', 0))
                    }
                
                logger.info(f"Loaded {len(food_groups)} food items from nutrition_data.csv")
                return food_groups
        except Exception as e:
            logger.error(f"Error loading nutrition_data.csv: {e}, using fallback data")
        
        # Final fallback to hardcoded data
        logger.warning("Using fallback hardcoded food groups data")
        return self._get_fallback_food_groups()
    
    def _get_fallback_food_groups(self):
        """Fallback food groups data if CSV file is not available."""
        return {
            'roti': {'protein': 3, 'carbs': 15, 'fats': 1, 'fiber': 2, 'calcium': 10, 'calories': 80},
            'rice': {'protein': 2, 'carbs': 28, 'fats': 0, 'fiber': 0, 'calcium': 1, 'calories': 130},
            'chicken': {'protein': 25, 'carbs': 0, 'fats': 3, 'fiber': 0, 'calcium': 10, 'calories': 165},
            'dal': {'protein': 8, 'carbs': 20, 'fats': 1, 'fiber': 5, 'calcium': 20, 'calories': 120},
            'paneer': {'protein': 18, 'carbs': 3, 'fats': 20, 'fiber': 0, 'calcium': 200, 'calories': 265},
            'sandwich': {'protein': 10, 'carbs': 30, 'fats': 8, 'fiber': 2, 'calcium': 50, 'calories': 250},
            'pudding': {'protein': 5, 'carbs': 45, 'fats': 12, 'fiber': 1, 'calcium': 80, 'calories': 320},
            'milk': {'protein': 8, 'carbs': 12, 'fats': 5, 'fiber': 0, 'calcium': 300, 'calories': 150},
            'egg': {'protein': 13, 'carbs': 1, 'fats': 11, 'fiber': 0, 'calcium': 50, 'calories': 155},
            'vegetable curry': {'protein': 4, 'carbs': 15, 'fats': 3, 'fiber': 6, 'calcium': 80, 'calories': 120}
        }
    
    def load_market_data(self):
        """Load market price data from INDB.csv or market_data.csv file with fallback to hardcoded data."""
        indb_csv_file = os.path.join(self.base_dir, 'INDB.csv')
        market_file = os.path.join(self.base_dir, 'market_data.csv')
        
        # Try loading from INDB.csv first (check if it has price data)
        try:
            if os.path.exists(indb_csv_file):
                df = pd.read_csv(indb_csv_file)
                
                if df is not None and not df.empty:
                    # Check for price-related columns in INDB.csv
                    price_cols = ['price', 'cost', 'avg_price', 'avg_price_per_100g', 'Price', 'Cost', 'market_price']
                    name_cols = ['food_name', 'food', 'item', 'name']
                    
                    price_col = next((col for col in price_cols if col in df.columns), None)
                    name_col = next((col for col in name_cols if col in df.columns), 'food_name')
                    
                    if price_col:
                        market_data = {}
                        for _, row in df.iterrows():
                            food_item = str(row[name_col]).lower().strip()
                            if pd.isna(food_item) or food_item == 'nan':
                                continue
                            price = float(row[price_col]) if pd.notna(row.get(price_col, 0)) else 50.0
                            market_data[food_item] = price
                        
                        if market_data:
                            logger.info(f"Loaded {len(market_data)} price entries from INDB.csv")
                            return market_data
        except Exception as e:
            logger.warning(f"Error loading prices from INDB.csv: {e}, trying market_data.csv fallback")
        
        # Fallback to market_data.csv file
        try:
            if os.path.exists(market_file):
                df = pd.read_csv(market_file)
                market_data = {}
                
                for _, row in df.iterrows():
                    food_item = str(row['food_item']).lower().strip()
                    # Use avg_price_per_100g from CSV, convert to price per item/100g
                    price = float(row.get('avg_price_per_100g', 50))
                    market_data[food_item] = price
                
                logger.info(f"Loaded {len(market_data)} price entries from market_data.csv")
                return market_data
        except Exception as e:
            logger.error(f"Error loading market_data.csv: {e}, using fallback data")
        
        # Final fallback to hardcoded data
        logger.warning("Using fallback hardcoded market data")
        return self._get_fallback_market_data()
    
    def _get_fallback_market_data(self):
        """Fallback market data if CSV file is not available."""
        return {
            'roti': 8, 'rice': 12, 'chicken': 60, 'dal': 15, 'paneer': 55,
            'bread': 5, 'milk': 6, 'egg': 8, 'potato': 8, 'tomato': 12,
            'sandwich': 25, 'pudding': 30, 'vegetable curry': 20
        }
    
    def get_food_item_data(self, item_name):
        """Get food data with fallback mappings."""
        original_name = item_name.lower()
        
        # Try exact match first
        if original_name in self.food_groups:
            return self.food_groups[original_name]
        
        # Try mapped name
        mapped_name = self.food_mappings.get(original_name)
        if mapped_name and mapped_name in self.food_groups:
            return self.food_groups[mapped_name]
        
        # Try partial matching
        for known_item in self.food_groups.keys():
            if known_item in original_name or original_name in known_item:
                return self.food_groups[known_item]
        
        # Final fallback - use average values
        return self.get_fallback_nutrient_profile()
    
    def get_fallback_nutrient_profile(self):
        """Return average nutrient profile for unknown items."""
        return {
            'protein': 8.0, 'carbs': 25.0, 'fats': 5.0, 
            'fiber': 2.0, 'calcium': 50.0, 'calories': 150
        }
    
    def validate_data_integrity(self):
        """Validate data integrity."""
        issues = []
        if not self.food_groups:
            issues.append("Food groups data not loaded")
        if not self.market_data:
            issues.append("Market data not loaded")
        return issues

# Enhanced Meal Economics
class MealEconomics:
    """
    Enhanced economic analysis with realistic pricing and ML insights.
    """
    
    def __init__(self, data_loader=None, ml_predictor=None):
        self.data_loader = data_loader
        self.ml_predictor = ml_predictor
        self.daily_food_allowance = 500  # Realistic daily allowance
        
    def set_daily_allowance(self, allowance: float):
        """Set the daily food allowance."""
        self.daily_food_allowance = allowance
        
    def calculate_meal_cost(self, food_name: str, quantity: float) -> float:
        """Calculate realistic cost for a food item with minimum Rs 30."""
        try:
            # Get prices from data_loader if available, otherwise use fallback
            if self.data_loader and hasattr(self.data_loader, 'market_data'):
                market_prices = self.data_loader.market_data
            else:
                # Fallback prices if data_loader is not available
                market_prices = {
                    'rice': 25.0, 'wheat': 30.0, 'bread': 35.0, 'chapati': 20.0,
                    'dosa': 40.0, 'idli': 25.0, 'roti': 18.0, 'naan': 45.0,
                    'paratha': 35.0, 'chicken': 120.0, 'fish': 150.0, 'mutton': 200.0,
                    'egg': 8.0, 'milk': 6.0, 'curd': 8.0, 'paneer': 60.0,
                    'cheese': 80.0, 'ghee': 120.0, 'oil': 80.0, 'butter': 100.0,
                    'apple': 60.0, 'banana': 20.0, 'orange': 40.0, 'mango': 50.0,
                    'tomato': 30.0, 'onion': 25.0, 'potato': 20.0, 'carrot': 35.0,
                    'spinach': 15.0, 'dal': 80.0, 'chana': 70.0, 'rajma': 90.0,
                    'biryani': 80.0
                }
            
            # Find the best matching price from CSV-loaded data
            food_lower = food_name.lower()
            base_price = 30.0  # Default minimum price
            
            # Try exact match first
            if food_lower in market_prices:
                base_price = market_prices[food_lower]
            else:
                # Try partial matching
                for key, price in market_prices.items():
                    if key in food_lower or food_lower in key:
                        base_price = price
                        break
            
            # Calculate cost based on quantity
            # Items that are sold per piece
            piece_items = ['roti', 'chapati', 'naan', 'paratha', 'puri', 'dosa', 'idli', 'egg']
            if any(item in food_lower for item in piece_items):
                # For these items, quantity is number of pieces
                cost = base_price * quantity
            else:
                # For other items, quantity is in grams, price is per 100g from CSV
                cost = (base_price * quantity) / 100
            
            # Ensure minimum cost of Rs 30 per food item
            return max(30.0, round(cost, 2))
            
        except Exception as e:
            logger.error(f"Error calculating meal cost for {food_name}: {e}")
            # Fallback with realistic minimum
            return 30.0
    
    def calculate_affordability_score(self, total_cost: float) -> float:
        """Calculate affordability score."""
        if self.daily_food_allowance <= 0:
            return 0.5
        score = 1 - (total_cost / self.daily_food_allowance)
        return max(0.0, min(1.0, score))
    
    def extract_nutrient_profile(self, food_name: str) -> Optional[List[float]]:
        """Extract nutrient profile for clustering."""
        if self.data_loader:
            data = self.data_loader.get_food_item_data(food_name)
            if data:
                return [
                    data.get('protein', 0),
                    data.get('carbs', 0),
                    data.get('fats', 0),
                    data.get('fiber', 0),
                    data.get('calcium', 0)
                ]
        return [8.0, 25.0, 5.0, 2.0, 50.0]  # Fallback profile
    
    def identify_expensive_ingredients(self, meal_items):
        """Identify the most expensive ingredients in current meals."""
        expensive_items = []
        
        # Get prices from data_loader if available
        if self.data_loader and hasattr(self.data_loader, 'market_data'):
            price_reference = self.data_loader.market_data
        else:
            # Fallback price reference data
            price_reference = {
                'chicken': 120, 'fish': 150, 'cheese': 80, 'almonds': 100,
                'bell pepper': 60, 'pasta': 40, 'bread': 35, 'lentils': 15,
                'potato': 20, 'rice': 25, 'egg': 8, 'spinach': 15,
                'paneer': 60, 'mutton': 200, 'butter': 100, 'ghee': 120
            }
        
        for item in meal_items:
            # Handle both string and dictionary items
            if isinstance(item, dict):
                item_name = item.get('food_name', '')
            else:
                item_name = str(item)
                
            item_lower = item_name.lower()
            # Try exact match first
            if item_lower in price_reference:
                expensive_items.append((item_lower, price_reference[item_lower]))
            else:
                # Try partial matching
                for ingredient, price in price_reference.items():
                    if ingredient in item_lower or item_lower in ingredient:
                        expensive_items.append((ingredient, price))
                        break
        
        return sorted(expensive_items, key=lambda x: x[1], reverse=True)
    
    def generate_nutrient_based_cost_savings(self, daily_snapshot) -> List[str]:
        """Generate specific cost-saving recommendations based on nutrient groups."""
        recommendations = []
        
        meal_items = getattr(daily_snapshot, 'meal_items', [])
        
        if meal_items:
            recommendations.append("---")
            recommendations.append("Cost Saving Recommendations:")
            
            # Check for expensive items and suggest alternatives
            expensive_ingredients = self.identify_expensive_ingredients(meal_items)
            
            for ingredient, cost in expensive_ingredients[:3]:
                recommendations.append(f"Consider alternatives for {ingredient} (Rs {cost:.2f})")
            
            # Nutrient-based alternatives
            recommendations.extend([
                "Proteins: Lentils (70% cheaper than chicken), Eggs (60% cheaper than fish)",
                "Carbs: Rice (50% cheaper than bread), Potatoes (80% cheaper than processed carbs)",
                "Vegetables: Seasonal local veggies (40-70% cheaper than exotic)",
                "Dairy: Curd instead of cheese (80% cheaper), Milk-based drinks vs packaged"
            ])
        
        return recommendations
    
    def generate_cost_insights(self, daily_snapshot, historical_meals=None) -> List[str]:
        """Generate comprehensive cost insights with nutrient-based recommendations."""
        insights = []
        
        total_cost = getattr(daily_snapshot, 'total_cost', 0)
        total_calories = getattr(daily_snapshot, 'total_calories', 0)
        
        # Enforce minimum cost threshold
        if total_cost < 30:
            insights.append("Cost Alert: Meal costs seem unusually low. Minimum expected cost: Rs 30")
            total_cost = max(total_cost, 30)
        
        if total_cost > 0:
            # Basic cost insight
            insights.append(f"Today's estimated meal cost: Rs {total_cost:.2f}")
            
            # Budget utilization
            budget_utilization = total_cost / self.daily_food_allowance
            insights.append(f"Budget utilization: {budget_utilization:.1%} of Rs {self.daily_food_allowance} allowance")
            
            # Affordability insight
            affordability = self.calculate_affordability_score(total_cost)
            if affordability > 0.7:
                insights.append("Excellent affordability within your budget")
            elif affordability > 0.4:
                insights.append("Moderate affordability - within budget but could optimize")
            else:
                insights.append("High cost alert - consider cost optimization strategies")
            
            # Cost efficiency
            if total_calories > 0:
                cost_per_calorie = total_cost / total_calories
                insights.append(f"Cost efficiency: Rs {cost_per_calorie:.4f}/calorie")
                
                if cost_per_calorie > 0.01:
                    insights.append("High cost per calorie - consider more efficient options")
                elif cost_per_calorie < 0.005:
                    insights.append("Excellent cost efficiency")
            
            # Generate nutrient-based cost saving recommendations
            cost_saving_tips = self.generate_nutrient_based_cost_savings(daily_snapshot)
            insights.extend(cost_saving_tips)
        
        # ML-powered insights if available
        if self.ml_predictor and historical_meals:
            try:
                ml_insights = self.ml_predictor.generate_ml_insights(
                    getattr(daily_snapshot, 'meal_items', []),
                    self.daily_food_allowance
                )
                insights.extend(ml_insights)
            except Exception as e:
                insights.append("ML Insights: Available with more data collection")
        
        return insights

# Enhanced ML Predictor
class MLPredictor:
    """Enhanced ML models with better clustering and decision trees."""
    
    def __init__(self):
        self.cost_regressor = LinearRegression()
        self.food_clusterer = KMeans(n_clusters=5, random_state=42)
        self.cost_efficiency_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.optimization_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def prepare_meal_features(self, meals_data):
        """Prepare features for ML models."""
        if not meals_data:
            return None, None, None
            
        features = []
        costs = []
        nutrition_data = []
        
        for meal in meals_data:
            if isinstance(meal, dict):
                calories = meal.get('calories', 0)
                protein = meal.get('protein_g', 0)
                carbs = meal.get('carbs_g', 0)
                fat = meal.get('fat_g', 0)
                quantity = meal.get('quantity', 100)
                cost = meal.get('cost', 0)
                
                feature_vector = [calories, protein, carbs, fat, quantity]
                features.append(feature_vector)
                costs.append(cost)
                nutrition_data.append({
                    'calories': calories,
                    'protein': protein,
                    'carbs': carbs,
                    'fat': fat,
                    'cost': cost,
                    'food_name': meal.get('food_name', 'Unknown')
                })
        
        return np.array(features), np.array(costs), nutrition_data
    
    def train_cost_prediction_model(self, meals_data):
        """Train regression model for cost prediction."""
        try:
            X, y, _ = self.prepare_meal_features(meals_data)
            
            if X is None or len(X) < 5:
                return {"status": "insufficient_data", "message": "Need at least 5 meals for training"}
            
            X_scaled = self.scaler.fit_transform(X)
            self.cost_regressor.fit(X_scaled, y)
            
            y_pred = self.cost_regressor.predict(X_scaled)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            self.is_trained = True
            
            return {
                "status": "success",
                "model_type": "LinearRegression",
                "training_samples": len(X),
                "mean_absolute_error": round(mae, 2),
                "r2_score": round(r2, 4),
                "feature_importance": ["calories", "protein", "carbs", "fat", "quantity"]
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def perform_nutrient_clustering(self, food_items, nutrient_data=None):
        """Perform nutrient-based clustering with descriptive labels."""
        try:
            if not food_items:
                return []
            
            # Prepare nutrient data if not provided
            if nutrient_data is None:
                nutrient_data = []
                for item in food_items:
                    # Mock nutrient extraction - replace with actual data
                    profile = [np.random.normal(10, 5), np.random.normal(25, 10), 
                              np.random.normal(8, 3), np.random.normal(2, 1), np.random.normal(50, 20)]
                    nutrient_data.append(profile)
            
            if len(nutrient_data) < 2:
                return [{"item": item, "nutrient_group": "Insufficient data", "group_number": -1} 
                       for item in food_items]
            
            # Perform clustering
            kmeans = KMeans(n_clusters=min(5, len(nutrient_data)), random_state=42)
            clusters = kmeans.fit_predict(nutrient_data)
            
            # Define descriptive cluster names based on centroid analysis
            cluster_names = self.analyze_cluster_centroids(kmeans.cluster_centers_)
            
            results = []
            for i, item in enumerate(food_items):
                cluster_num = clusters[i]
                results.append({
                    "item": item,
                    "nutrient_group": cluster_names[cluster_num],
                    "group_number": int(cluster_num),
                    "nutrient_profile": nutrient_data[i]
                })
            
            return results
            
        except Exception as e:
            print(f"Clustering error: {e}")
            return [{"item": item, "nutrient_group": "Clustering failed", "group_number": -1} 
                   for item in food_items]
    
    def analyze_cluster_centroids(self, centroids):
        """Analyze cluster centroids to assign descriptive names."""
        nutrient_features = ['protein', 'carbs', 'fats', 'fiber', 'calcium']
        cluster_names = []
        
        for i, centroid in enumerate(centroids):
            # Find dominant nutrients in this cluster
            nutrient_scores = list(zip(nutrient_features, centroid))
            nutrient_scores.sort(key=lambda x: x[1], reverse=True)
            
            top_nutrients = [nutrient for nutrient, score in nutrient_scores[:2]]
            
            # Assign descriptive name based on dominant nutrients
            if 'protein' in top_nutrients and 'calcium' in top_nutrients:
                name = "High Protein & Calcium"
            elif 'protein' in top_nutrients and 'fats' in top_nutrients:
                name = "Protein & Healthy Fats"
            elif 'carbs' in top_nutrients and 'protein' in top_nutrients:
                name = "Balanced Energy & Protein"
            elif 'carbs' in top_nutrients:
                name = "Energy Rich Carbs"
            elif 'fats' in top_nutrients:
                name = "Healthy Fats & Oils"
            elif 'fiber' in top_nutrients:
                name = "High Fiber"
            else:
                name = f"Nutrition Group {i+1}"
            
            cluster_names.append(name)
        
        return cluster_names
    
    def cluster_foods_by_nutrition(self, meals_data):
        """Cluster food items based on nutrition profile."""
        try:
            X, _, nutrition_data = self.prepare_meal_features(meals_data)
            
            if X is None or len(X) < 3:
                return {"status": "insufficient_data"}
            
            food_items = [item['food_name'] for item in nutrition_data]
            nutrition_features = X[:, :4]  # Use calories, protein, carbs, fat
            
            # Perform enhanced clustering
            clustering_results = self.perform_nutrient_clustering(food_items, nutrition_features)
            
            # Format results
            results = {
                "status": "success",
                "algorithm": "Enhanced KMeans",
                "n_clusters": len(set([r['group_number'] for r in clustering_results if r['group_number'] != -1])),
                "clustering_results": clustering_results
            }
            
            return results
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def train_cost_efficiency_model(self, meals_data):
        """Train decision tree for cost efficiency analysis."""
        try:
            X, y, nutrition_data = self.prepare_meal_features(meals_data)
            
            if X is None or len(X) < 8:
                return {"status": "insufficient_data"}
            
            # Calculate cost efficiency (calories per rupee)
            cost_efficiency = []
            for i in range(len(X)):
                calories = X[i][0]
                cost = y[i] if i < len(y) else 50
                if cost > 0:
                    efficiency = calories / cost
                else:
                    efficiency = 0
                cost_efficiency.append(efficiency)
            
            X_nutrition = X[:, :4]  # Use nutrition features only
            self.cost_efficiency_tree.fit(X_nutrition, cost_efficiency)
            
            importance = self.cost_efficiency_tree.feature_importances_
            feature_names = ['calories', 'protein', 'carbs', 'fat']
            
            return {
                "status": "success",
                "model_type": "DecisionTreeRegressor",
                "training_samples": len(X),
                "feature_importance": dict(zip(feature_names, [round(float(imp), 4) for imp in importance])),
                "target": "cost_efficiency"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def build_cost_optimization_tree(self):
        """Build decision tree for cost optimization recommendations."""
        # Training data: [protein_rich, carb_rich, budget_tight, high_calorie_needed -> recommendation]
        training_data = [
            [1, 1, 0, 1, 'balanced_meals'],      # Normal needs, normal budget
            [1, 0, 0, 0, 'protein_focus'],       # High protein need
            [0, 1, 0, 1, 'energy_focus'],        # High energy need  
            [1, 1, 1, 0, 'budget_protein'],      # Tight budget, need protein
            [0, 1, 1, 1, 'budget_energy'],       # Tight budget, need energy
            [0, 0, 1, 0, 'minimal_cost']         # Minimal requirements
        ]
        
        X = [item[:4] for item in training_data]
        y = [item[4] for item in training_data]
        
        tree = DecisionTreeClassifier(random_state=42, max_depth=3)
        tree.fit(X, y)
        
        return tree
    
    def get_cost_optimization_decision(self, protein_rich=True, carb_rich=True, 
                                     budget_tight=False, high_calorie_needed=True):
        """Get cost optimization decision based on current needs."""
        tree = self.build_cost_optimization_tree()
        
        features = [
            1 if protein_rich else 0,
            1 if carb_rich else 0, 
            1 if budget_tight else 0,
            1 if high_calorie_needed else 0
        ]
        
        decision = tree.predict([features])[0]
        
        # Decision explanations
        explanations = {
            'balanced_meals': "Opt for balanced meals with mix of protein and carbs",
            'protein_focus': "Focus on cost-effective protein sources like lentils and eggs",
            'energy_focus': "Prioritize energy-rich foods like rice and potatoes",
            'budget_protein': "Use plant-based proteins and seasonal vegetables",
            'budget_energy': "Choose rice, potatoes and local seasonal produce",
            'minimal_cost': "Minimal cost approach with basic staples"
        }
        
        return {
            'decision': decision,
            'explanation': explanations.get(decision, "Consider balanced meal options"),
            'features_used': ['protein_rich', 'carb_rich', 'budget_tight', 'high_calorie_needed']
        }
    
    def predict_meal_cost(self, nutrition_info):
        """Predict meal cost using regression with minimum threshold."""
        if not self.is_trained:
            # Fallback prediction with minimum cost
            base_cost = 30.0
            calories = nutrition_info.get('calories', 0)
            if calories > 0:
                base_cost += calories * 0.01  # Basic scaling
            return max(30.0, base_cost)
            
        try:
            features = np.array([
                nutrition_info.get('calories', 0),
                nutrition_info.get('protein_g', 0),
                nutrition_info.get('carbs_g', 0),
                nutrition_info.get('fat_g', 0),
                nutrition_info.get('quantity', 100)
            ]).reshape(1, -1)
            
            features_scaled = self.scaler.transform(features)
            prediction = self.cost_regressor.predict(features_scaled)
            
            return max(30.0, round(float(prediction[0]), 2))
            
        except Exception as e:
            return 30.0  # Reasonable fallback
    
    def find_cost_effective_alternatives(self, current_meal, all_meals, threshold=0.2):
        """Find cost-effective alternatives using ML analysis."""
        alternatives = []
        
        if not all_meals:
            return alternatives
        
        try:
            current_cost = current_meal.get('cost', 0)
            current_calories = current_meal.get('calories', 0)
            current_protein = current_meal.get('protein_g', 0)
            
            if current_cost <= 0:
                return alternatives
            
            for meal in all_meals:
                if meal.get('food_name') != current_meal.get('food_name'):
                    alt_cost = meal.get('cost', 0)
                    alt_calories = meal.get('calories', 0)
                    alt_protein = meal.get('protein_g', 0)
                    
                    # Only consider if alternative is cheaper
                    if alt_cost < current_cost:
                        cost_reduction = (current_cost - alt_cost) / current_cost
                        calorie_similarity = abs(current_calories - alt_calories) / max(current_calories, 1)
                        protein_similarity = abs(current_protein - alt_protein) / max(current_protein, 1)
                        
                        # More flexible matching for better recommendations
                        if (cost_reduction >= threshold and 
                            calorie_similarity <= 0.3 and  # 30% calorie difference allowed
                            protein_similarity <= 0.4):    # 40% protein difference allowed
                            
                            alternatives.append({
                                'food_name': meal.get('food_name'),
                                'current_cost': current_cost,
                                'alternative_cost': alt_cost,
                                'cost_saving': current_cost - alt_cost,
                                'cost_reduction_pct': round(cost_reduction * 100, 1),
                                'calories': alt_calories,
                                'protein': alt_protein
                            })
            
            alternatives.sort(key=lambda x: x['cost_saving'], reverse=True)
            return alternatives[:5]  # Return top 5 alternatives
            
        except Exception as e:
            return []
    
    def generate_ml_insights(self, meals_data, daily_allowance=500):
        """Generate ML-powered insights for cost optimization."""
        insights = []
        
        if not meals_data:
            return ["Log more meals to get personalized insights"]
        
        try:
            total_cost = sum(meal.get('cost', 0) for meal in meals_data)
            total_calories = sum(meal.get('calories', 0) for meal in meals_data)
            total_protein = sum(meal.get('protein_g', 0) for meal in meals_data)
            
            # Cost efficiency insight
            if total_cost > 0 and total_calories > 0:
                cost_per_calorie = total_cost / total_calories
                if cost_per_calorie > 0.015:
                    insights.append("High spending detected: Consider more cost-effective foods")
                elif cost_per_calorie < 0.005:
                    insights.append("Excellent value: Great cost efficiency")
                else:
                    insights.append("Good balance: Reasonable cost for nutrition")
            
            # Budget utilization insight
            if daily_allowance > 0:
                budget_utilization = (total_cost / daily_allowance) * 100
                if budget_utilization > 90:
                    insights.append(f"Budget alert: Using {budget_utilization:.0f}% of daily allowance")
                elif budget_utilization < 40:
                    insights.append(f"Under budget: Only {budget_utilization:.0f}% of allowance used")
            
            # Protein cost efficiency
            protein_meals = [meal for meal in meals_data if meal.get('protein_g', 0) > 10]
            if protein_meals:
                protein_costs = []
                for meal in protein_meals:
                    protein = meal.get('protein_g', 0)
                    cost = meal.get('cost', 0)
                    if protein > 0 and cost > 0:
                        protein_costs.append(cost / protein)
                
                if protein_costs:
                    avg_protein_cost = np.mean(protein_costs)
                    if avg_protein_cost > 6:
                        insights.append("Protein cost: Consider plant-based proteins for savings")
                    elif avg_protein_cost < 3:
                        insights.append("Great protein value: Efficient protein sources")
            
            # Decision tree recommendation
            if len(meals_data) >= 3:
                decision_data = self.get_cost_optimization_decision(
                    protein_rich=total_protein > 40,
                    carb_rich=total_calories > 1500,
                    budget_tight=total_cost > daily_allowance * 0.7,
                    high_calorie_needed=total_calories < 1800
                )
                insights.append(f"Smart suggestion: {decision_data['explanation']}")
            
            return insights
            
        except Exception as e:
            return ["Analysis: Processing insights with available data"]

# Meal Aggregator for Daily Analysis
class MealAggregator:
    """Aggregate meal data for daily analysis."""
    
    def __init__(self):
        pass
    
    def create_daily_snapshot(self, meals_data, date_obj):
        """Create daily snapshot from meal data."""
        snapshot = type('DailySnapshot', (), {})()
        
        # Initialize totals
        snapshot.total_cost = 0
        snapshot.total_calories = 0
        snapshot.total_protein = 0
        snapshot.total_carbs = 0
        snapshot.total_fat = 0
        snapshot.meal_items = []
        snapshot.category_costs = {}
        snapshot.date = date_obj
        
        # Aggregate data
        for meal in meals_data:
            if isinstance(meal, dict):
                snapshot.total_cost += meal.get('cost', 0)
                snapshot.total_calories += meal.get('calories', 0)
                snapshot.total_protein += meal.get('protein_g', 0)
                snapshot.total_carbs += meal.get('carbs_g', 0)
                snapshot.total_fat += meal.get('fat_g', 0)
                
                # Add to meal items
                snapshot.meal_items.append({
                    'food_name': meal.get('food_name', 'Unknown'),
                    'calories': meal.get('calories', 0),
                    'protein_g': meal.get('protein_g', 0),
                    'carbs_g': meal.get('carbs_g', 0),
                    'fat_g': meal.get('fat_g', 0),
                    'cost': meal.get('cost', 0),
                    'quantity': meal.get('quantity', 100)
                })
        
        # Calculate derived metrics
        snapshot.affordability_score = 1 - (snapshot.total_cost / 500) if snapshot.total_cost > 0 else 1.0
        snapshot.cost_per_calorie = snapshot.total_cost / snapshot.total_calories if snapshot.total_calories > 0 else 0
        
        # Categorize costs (simplified)
        food_categories = {
            'Proteins': ['chicken', 'fish', 'egg', 'paneer', 'dal', 'lentils'],
            'Carbs': ['rice', 'roti', 'bread', 'dosa', 'poha'],
            'Vegetables': ['potato', 'tomato', 'onion', 'carrot', 'spinach'],
            'Dairy': ['milk', 'curd', 'cheese', 'butter'],
            'Other': []
        }
        
        for category, keywords in food_categories.items():
            category_cost = 0
            for meal in meals_data:
                food_name = meal.get('food_name', '').lower()
                if any(keyword in food_name for keyword in keywords):
                    category_cost += meal.get('cost', 0)
                elif category == 'Other' and not any(any(kw in food_name for kw in kw_list) for kw_list in food_categories.values() if kw_list != []):
                    category_cost += meal.get('cost', 0)
            
            if category_cost > 0:
                snapshot.category_costs[category] = category_cost
        
        return snapshot
    
    def get_cost_analysis(self, meals_data, days=30):
        """Get cost analysis over time."""
        # Group by date
        date_meals = {}
        for meal in meals_data:
            if isinstance(meal, dict):
                meal_date = meal.get('date', date.today())
                if meal_date not in date_meals:
                    date_meals[meal_date] = []
                date_meals[meal_date].append(meal)
        
        # Create daily snapshots
        daily_snapshots = []
        for meal_date, daily_meals in date_meals.items():
            snapshot = self.create_daily_snapshot(daily_meals, meal_date)
            daily_snapshots.append({
                'date': meal_date,
                'total_cost': snapshot.total_cost,
                'total_calories': snapshot.total_calories,
                'total_protein': snapshot.total_protein,
                'total_carbs': snapshot.total_carbs,
                'total_fat': snapshot.total_fat,
                'affordability_score': snapshot.affordability_score
            })
        
        return {
            'daily_snapshots': daily_snapshots,
            'analysis_period': days,
            'total_meals_analyzed': len(meals_data)
        }

# Enhanced Storage with Sample Data and Calendar Support
class EnhancedStorage:
    def __init__(self):
        self.meals = self.initialize_sample_data()
        
    def initialize_sample_data(self):
        """Initialize with sample meals for the last 30 days."""
        sample_meals = []
        
        # Common Indian foods that exist in our database
        common_foods = [
            {"name": "roti", "calories": 80, "protein": 3, "carbs": 15, "fat": 1, "cost": 50},
            {"name": "rice", "calories": 130, "protein": 2, "carbs": 28, "fat": 0, "cost": 60},
            {"name": "dal", "calories": 120, "protein": 8, "carbs": 20, "fat": 1, "cost": 55},
            {"name": "vegetable curry", "calories": 120, "protein": 4, "carbs": 15, "fat": 3, "cost": 65},
            {"name": "chicken curry", "calories": 165, "protein": 25, "carbs": 0, "fat": 3, "cost": 120},
            {"name": "milk", "calories": 150, "protein": 8, "carbs": 12, "fat": 5, "cost": 50},
            {"name": "egg", "calories": 155, "protein": 13, "carbs": 1, "fat": 11, "cost": 55},
            {"name": "paneer", "calories": 265, "protein": 18, "carbs": 3, "fat": 20, "cost": 110}
        ]
        
        # Generate sample data for last 30 days
        for i in range(30):
            meal_date = date.today() - timedelta(days=29-i)
            
            # Add 2-4 meals per day
            num_meals = np.random.randint(2, 5)
            for j in range(num_meals):
                food = np.random.choice(common_foods)
                sample_meals.append({
                    "date": meal_date,
                    "food_name": food["name"],
                    "cuisine": "Indian",
                    "carbs_g": food["carbs"],
                    "protein_g": food["protein"],
                    "fat_g": food["fat"],
                    "calories": food["calories"],
                    "quantity": 100,
                    "unit": "g",
                    "original_input": f"100g {food['name']}",
                    "matched_from_db": True,
                    "cost": food["cost"]
                })
        
        return sample_meals
        
    def get_meals_for_date(self, dt):
        date_meals = [meal for meal in self.meals if meal.get('date') == dt]
        return pd.DataFrame(date_meals) if date_meals else pd.DataFrame()
    
    def get_recent_meals(self, days=30):
        cutoff_date = date.today() - timedelta(days=days)
        recent_meals = [meal for meal in self.meals 
                      if meal.get('date', date.today()) >= cutoff_date]
        return pd.DataFrame(recent_meals) if recent_meals else pd.DataFrame()
    
    def add_meal_entry(self, entry):
        self.meals.append(entry)
    
    def update_meal_entry(self, index, updated_meal):
        """Update a meal entry by index."""
        if 0 <= index < len(self.meals):
            self.meals[index] = updated_meal
    
    def delete_meal_entry(self, index):
        """Delete a meal entry by index."""
        if 0 <= index < len(self.meals):
            del self.meals[index]
    
    def get_meals_by_date_range(self, start_date, end_date):
        """Get meals for a date range."""
        range_meals = [meal for meal in self.meals 
                      if start_date <= meal.get('date') <= end_date]
        return pd.DataFrame(range_meals) if range_meals else pd.DataFrame()
    
    def get_most_frequent_meals(self, days=30, top_n=5):
        """Get most frequently consumed meals in the last N days."""
        recent_meals_df = self.get_recent_meals(days)
        
        if recent_meals_df.empty:
            return []
        
        # Group meals by date to find common combinations
        meals_by_date = {}
        for _, meal in recent_meals_df.iterrows():
            meal_date = meal['date']
            if meal_date not in meals_by_date:
                meals_by_date[meal_date] = []
            meals_by_date[meal_date].append(meal['food_name'].lower())
        
        # Create meal combination patterns
        meal_patterns = {}
        for meal_date, foods in meals_by_date.items():
            # Check for common meal combinations
            foods_set = set(foods)
            
            # Roti + vegetable curry pattern
            if 'roti' in foods_set and any('vegetable' in f or 'curry' in f or 'sabji' in f for f in foods_set):
                pattern = ("2 Roti + Sabji", "2 roti, 1 bowl vegetable curry")
                meal_patterns[pattern] = meal_patterns.get(pattern, 0) + 1
            
            # Rice + dal pattern
            if 'rice' in foods_set and 'dal' in foods_set:
                pattern = ("Rice + Dal", "1 plate rice, 1 bowl dal")
                meal_patterns[pattern] = meal_patterns.get(pattern, 0) + 1
            
            # Chicken curry pattern
            if 'chicken' in foods_set:
                pattern = ("Chicken Curry", "200g chicken curry")
                meal_patterns[pattern] = meal_patterns.get(pattern, 0) + 1
            
            # Paneer pattern
            if 'paneer' in foods_set:
                pattern = ("Paneer Dish", "150g paneer sabji")
                meal_patterns[pattern] = meal_patterns.get(pattern, 0) + 1
        
        # Also get individual food frequencies
        food_counts = recent_meals_df['food_name'].str.lower().value_counts()
        
        # If no patterns found, use individual frequent foods
        if not meal_patterns:
            frequent_items = []
            for food_name, count in food_counts.head(10).items():
                if food_name == 'roti':
                    frequent_items.append(("2 Roti + Sabji", "2 roti, 1 bowl vegetable curry", count))
                elif food_name == 'rice':
                    frequent_items.append(("Rice + Dal", "1 plate rice, 1 bowl dal", count))
                elif food_name == 'chicken':
                    frequent_items.append(("Chicken Curry", "200g chicken curry", count))
                elif food_name == 'paneer':
                    frequent_items.append(("Paneer Dish", "150g paneer sabji", count))
            
            frequent_items.sort(key=lambda x: x[2], reverse=True)
            return [(name, input_text) for name, input_text, _ in frequent_items[:top_n]]
        
        # Sort patterns by frequency
        sorted_patterns = sorted(meal_patterns.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N most frequent patterns
        return [(name, input_text) for (name, input_text), _ in sorted_patterns[:top_n]]

# Fallback implementations for missing modules
class FallbackNutritionDB:
    def parse_and_get_nutrition(self, text):
        items = [item.strip() for item in text.split(',')]
        found_items = []
        for item in items:
            # Map common foods to known items
            food_mapping = {
                'roti': {'calories': 80, 'protein_g': 3, 'carbs_g': 15, 'fat_g': 1},
                'rice': {'calories': 130, 'protein_g': 2, 'carbs_g': 28, 'fat_g': 0},
                'dal': {'calories': 120, 'protein_g': 8, 'carbs_g': 20, 'fat_g': 1},
                'milk': {'calories': 150, 'protein_g': 8, 'carbs_g': 12, 'fat_g': 5},
                'egg': {'calories': 155, 'protein_g': 13, 'carbs_g': 1, 'fat_g': 11},
                'chicken': {'calories': 165, 'protein_g': 25, 'carbs_g': 0, 'fat_g': 3},
                'paneer': {'calories': 265, 'protein_g': 18, 'carbs_g': 3, 'fat_g': 20},
                'vegetable': {'calories': 120, 'protein_g': 4, 'carbs_g': 15, 'fat_g': 3}
            }
            
            nutrients = {'calories': 150, 'protein_g': 8, 'carbs_g': 25, 'fat_g': 5}  # Default
            for known_food, known_nutrients in food_mapping.items():
                if known_food in item.lower():
                    nutrients = known_nutrients.copy()
                    break
            
            found_items.append({
                'matched_name': item.strip(),
                'calories': nutrients['calories'],
                'protein_g': nutrients['protein_g'],
                'carbs_g': nutrients['carbs_g'],
                'fat_g': nutrients['fat_g'],
                'quantity': 100,
                'unit': 'g',
                'original': item.strip()
            })
        return {'found_items': found_items, 'missing_items': []}
    
    def add_missing_food(self, *args, **kwargs):
        pass

def get_rda_for_user(age, gender):
    return {'calories': 2000, 'protein_g': 50}

# Initialize global instances
def get_data_loader():
    return EnhancedDataLoader()

def get_meal_economics():
    return MealEconomics()

def get_meal_aggregator():
    return MealAggregator()

def get_ml_predictor():
    return MLPredictor()

# Main Streamlit UI Class
class MealCostOptimizerUI:
    """Enhanced Streamlit UI for Meal Cost Optimization."""
    
    def __init__(self):
        self.components = self.initialize_components()
        self.current_analysis = None
        
    def initialize_components(self):
        """Initialize all components with ML integration."""
        data_loader = get_data_loader()
        ml_predictor = get_ml_predictor()
        meal_economics = get_meal_economics()
        meal_aggregator = get_meal_aggregator()
        
        # Set up dependencies
        meal_economics.data_loader = data_loader
        meal_economics.ml_predictor = ml_predictor
        
        return {
            'data_loader': data_loader,
            'meal_economics': meal_economics,
            'meal_aggregator': meal_aggregator,
            'ml_predictor': ml_predictor
        }
    
    def render_calendar_tab(self):
        """Render calendar tab for viewing and editing meals."""
        st.header("Calendar - View and Edit Meals")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", value=date.today())
        
        # Get meals for date range
        range_meals_df = indb_store.get_meals_by_date_range(start_date, end_date)
        
        if not range_meals_df.empty:
            st.subheader(f"Meals from {start_date} to {end_date}")
            
            # Group by date
            meals_by_date = {}
            for _, meal in range_meals_df.iterrows():
                meal_date = meal['date']
                if meal_date not in meals_by_date:
                    meals_by_date[meal_date] = []
                meals_by_date[meal_date].append(meal)
            
            # Display meals by date
            for meal_date in sorted(meals_by_date.keys(), reverse=True):
                with st.expander(f"{meal_date} - {len(meals_by_date[meal_date])} meals"):
                    daily_meals = meals_by_date[meal_date]
                    total_cost = sum(meal['cost'] for meal in daily_meals)
                    total_calories = sum(meal['calories'] for meal in daily_meals)
                    
                    st.write(f"Total: Rs {total_cost:.2f}, {total_calories} calories")
                    
                    for i, meal in enumerate(daily_meals):
                        col1, col2, col3 = st.columns([3, 2, 1])
                        with col1:
                            st.write(f"{meal['food_name']} - {meal['quantity']}g")
                        with col2:
                            st.write(f"Rs {meal['cost']:.2f}, {meal['calories']} cal")
                        with col3:
                            if st.button("Delete", key=f"del_{meal_date}_{i}"):
                                # Find and delete the meal
                                for idx, stored_meal in enumerate(indb_store.meals):
                                    if (stored_meal['date'] == meal_date and 
                                        stored_meal['food_name'] == meal['food_name'] and
                                        stored_meal['cost'] == meal['cost']):
                                        indb_store.delete_meal_entry(idx)
                                        st.rerun()
                                        break
            
            # Summary statistics
            st.subheader("Period Summary")
            total_days = (end_date - start_date).days + 1
            total_meals = len(range_meals_df)
            avg_meals_per_day = total_meals / total_days
            avg_cost_per_day = range_meals_df['cost'].sum() / total_days
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Days", total_days)
            with col2:
                st.metric("Total Meals", total_meals)
            with col3:
                st.metric("Avg Meals/Day", f"{avg_meals_per_day:.1f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Cost", f"Rs {range_meals_df['cost'].sum():.2f}")
            with col2:
                st.metric("Avg Daily Cost", f"Rs {avg_cost_per_day:.2f}")
        
        else:
            st.info("No meals found in the selected date range")
    
    def render_cost_recommendations_tab(self):
        """Render the enhanced cost saving recommendations tab."""
        st.header("Cost Saving Recommendations")
        
        # Always show recommendations, even without current analysis
        if not hasattr(self, 'current_analysis') or not self.current_analysis:
            # Show general recommendations
            st.info("General Cost Saving Tips:")
            
            st.subheader("Protein Sources")
            st.write("Budget (Rs 15-30): Lentils, Eggs, Chickpeas, Paneer")
            st.write("Moderate (Rs 40-80): Chicken, Fish, Tofu")
            st.write("Premium (Rs 80-150): Lean Meat, Imported Cheese, Salmon")
            
            st.subheader("Carbohydrate Sources")
            st.write("Budget (Rs 10-20): Rice, Potatoes, Whole Wheat Flour")
            st.write("Moderate (Rs 25-40): Oats, Multigrain Bread, Pasta")
            st.write("Premium (Rs 50-100): Quinoa, Brown Rice, Specialty Grains")
            
            st.subheader("Vegetables & Fruits")
            st.write("Budget (Rs 15-30): Seasonal Local Vegetables, Potatoes, Onions")
            st.write("Moderate (Rs 40-60): Bell Peppers, Broccoli, Cauliflower")
            st.write("Premium (Rs 70-120): Imported Vegetables, Organic Produce")
            
            return
        
        # Generate and display recommendations for current analysis
        historical_meals_df = indb_store.get_recent_meals(30)
        historical_meals = historical_meals_df.to_dict('records') if not historical_meals_df.empty else []
        
        insights = self.components['meal_economics'].generate_cost_insights(
            self.current_analysis, historical_meals
        )
        
        for insight in insights:
            if "Cost Saving Recommendations" in insight or "Consider alternatives" in insight:
                st.success(insight)
            elif "Cost Alert" in insight or "High cost alert" in insight or "alert" in insight.lower():
                st.error(insight)
            elif "Excellent" in insight.lower() or "efficient" in insight.lower():
                st.success(insight)
            elif "consider" in insight.lower() or "High cost" in insight:
                st.warning(insight)
            else:
                st.info(insight)
        
        # Show nutrient-based alternatives
        st.subheader("Nutrient-Based Cost Saving Alternatives")
        
        nutrient_alternatives = {
            "Protein Sources": [
                "Budget (Rs 15-30): Lentils, Eggs, Chickpeas, Paneer",
                "Moderate (Rs 40-80): Chicken, Fish, Tofu", 
                "Premium (Rs 80-150): Lean Meat, Imported Cheese, Salmon"
            ],
            "Carbohydrate Sources": [
                "Budget (Rs 10-20): Rice, Potatoes, Whole Wheat Flour",
                "Moderate (Rs 25-40): Oats, Multigrain Bread, Pasta",
                "Premium (Rs 50-100): Quinoa, Brown Rice, Specialty Grains"
            ],
            "Vegetables & Fruits": [
                "Budget (Rs 15-30): Seasonal Local Vegetables, Potatoes, Onions",
                "Moderate (Rs 40-60): Bell Peppers, Broccoli, Cauliflower",
                "Premium (Rs 70-120): Imported Vegetables, Organic Produce"
            ]
        }
        
        for category, options in nutrient_alternatives.items():
            with st.expander(f"{category} Options"):
                for option in options:
                    st.write(option)
        
        # Decision Tree Recommendations
        st.subheader("Smart Optimization Suggestions")
        
        if hasattr(self, 'current_analysis') and self.current_analysis:
            total_calories = getattr(self.current_analysis, 'total_calories', 0)
            total_protein = getattr(self.current_analysis, 'total_protein', 0)
            total_cost = getattr(self.current_analysis, 'total_cost', 0)
            
            decision = self.components['ml_predictor'].get_cost_optimization_decision(
                protein_rich=total_protein > 30,
                carb_rich=total_calories > 1200,
                budget_tight=total_cost > 400,
                high_calorie_needed=total_calories < 1800
            )
            
            st.info(f"Optimization Strategy: {decision['explanation']}")
    
    def render_ml_analysis_tab(self):
        """Render enhanced ML analysis tab."""
        st.header("Nutrition Analysis")
        
        # Get historical data for ML
        historical_meals_df = indb_store.get_recent_meals(30)
        
        if not historical_meals_df.empty:
            historical_meals = historical_meals_df.to_dict('records')
            
            # Always show ML features, even without current analysis
            st.subheader("Model Training")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Train Cost Prediction"):
                    with st.spinner("Training regression model..."):
                        result = self.components['ml_predictor'].train_cost_prediction_model(historical_meals)
                        if result["status"] == "success":
                            st.success("Regression model trained!")
                            st.write(f"Mean Absolute Error: Rs {result['mean_absolute_error']}")
                            st.write(f"R2 Score: {result['r2_score']}")
                        else:
                            st.error(f"Training failed: {result.get('message', 'Unknown error')}")
            
            with col2:
                if st.button("Cluster Foods"):
                    with st.spinner("Clustering food items by nutrition..."):
                        result = self.components['ml_predictor'].cluster_foods_by_nutrition(historical_meals)
                        if result["status"] == "success":
                            st.success("Food clustering completed!")
                            
                            # Display clustering results
                            if "clustering_results" in result:
                                clusters = {}
                                for item in result["clustering_results"]:
                                    group = item.get('nutrient_group', 'Unknown')
                                    if group not in clusters:
                                        clusters[group] = []
                                    clusters[group].append(item['item'])
                                
                                for group_name, items in clusters.items():
                                    with st.expander(f"{group_name} ({len(items)} items)"):
                                        for item in items:
                                            st.write(f"{item}")
            
            with col3:
                if st.button("Train Efficiency Model"):
                    with st.spinner("Training decision tree for cost efficiency..."):
                        result = self.components['ml_predictor'].train_cost_efficiency_model(historical_meals)
                        if result["status"] == "success":
                            st.success("Efficiency model trained!")
                            st.write("Feature Importance:")
                            st.json(result["feature_importance"])
            
            # Display clustering results for today's meals if available
            if hasattr(self, 'current_analysis') and self.current_analysis:
                st.subheader("Nutrient Group Clustering")
                
                meal_items = getattr(self.current_analysis, 'meal_items', [])
                if meal_items:
                    food_names = [item.get('food_name', 'Unknown') for item in meal_items]
                    
                    clustering_result = self.components['ml_predictor'].perform_nutrient_clustering(food_names)
                    
                    if clustering_result:
                        # Group by cluster
                        clusters = {}
                        for result in clustering_result:
                            group_name = result.get('nutrient_group', 'Unknown')
                            if group_name not in clusters:
                                clusters[group_name] = []
                            clusters[group_name].append(result['item'])
                        
                        for group_name, items in clusters.items():
                            with st.expander(f"{group_name} ({len(items)} items)"):
                                for item in items:
                                    st.write(f"{item}")
            
            # Cost-effective alternatives
            st.subheader("Cost-Saving Alternatives")
            
            today_meals = getattr(self.current_analysis, 'meal_items', []) if hasattr(self, 'current_analysis') else []
            if today_meals and historical_meals:
                alternatives_found = False
                for meal in today_meals:
                    alternatives = self.components['ml_predictor'].find_cost_effective_alternatives(
                        meal, historical_meals
                    )
                    if alternatives:
                        alternatives_found = True
                        st.write(f"Alternatives for {meal.get('food_name', 'Unknown')}:")
                        for alt in alternatives:
                            st.success(
                                f"{alt['food_name']}: Save Rs {alt['cost_saving']:.2f} "
                                f"({alt['cost_reduction_pct']}%) - {alt['calories']} cal, {alt['protein']}g protein"
                            )
                
                if not alternatives_found and today_meals:
                    st.info("Your current choices are cost-effective.")
        
        else:
            st.info("Sample data loaded. Log more meals to enable advanced ML analysis features")
    
    def render(self):
        """Main render function for the Streamlit app."""
        # Set page config
        st.set_page_config(
            page_title="Enhanced Meal Cost Optimizer",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Main title
        st.title("Foodnomics")
        st.markdown("Spend less, eat better")
        
        # Initialize enhanced storage and nutrition DB
        global indb_nutrition_db, parse_and_get_nutrition, indb_store
        indb_nutrition_db = FallbackNutritionDB()
        parse_and_get_nutrition = indb_nutrition_db.parse_and_get_nutrition
        indb_store = EnhancedStorage()  # Use enhanced storage with sample data
        
        # Sidebar for user settings
        with st.sidebar:
            st.header("Settings")
            
            # User profile
            st.subheader("Profile")
            age = st.number_input("Age", min_value=1, max_value=120, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            
            # Food allowance setting
            st.subheader("Daily Food Allowance")
            use_custom_allowance = st.checkbox("Use custom allowance", value=False)
            if use_custom_allowance:
                daily_allowance = st.number_input(
                    "Daily Food Allowance (Rs)", 
                    min_value=50,
                    max_value=5000, 
                    value=500,
                    help="Your daily budget for food"
                )
                self.components['meal_economics'].set_daily_allowance(daily_allowance)
            else:
                daily_allowance = 500
                st.info(f"Using default allowance: Rs {daily_allowance}/day")
            
            # RDA calculation
            rda = get_rda_for_user(age, gender)
            st.caption(f"Daily RDA: {rda['calories']} kcal, {rda['protein_g']}g protein")
            
            # Data validation
            st.subheader("Data Status")
            issues = self.components['data_loader'].validate_data_integrity()
            if issues:
                st.error("Data issues found:")
                for issue in issues:
                    st.error(f"{issue}")
            else:
                st.success("All data files valid")
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Log Meals", 
            "Daily Snapshot", 
            "Periodic Trends", 
            "Pattern Analysis",
            "Price Recommendations",
            "Calendar"
        ])
        
        with tab1:
            self.render_meal_logging_tab()
        
        with tab2:
            self.render_today_insights_tab()
        
        with tab3:
            self.render_trends_tab()
        
        with tab4:
            self.render_ml_analysis_tab()
        
        with tab5:
            self.render_cost_recommendations_tab()
        
        with tab6:
            self.render_calendar_tab()
        
        # Footer
        st.markdown("---")
        st.markdown("Based on the INDB and Market Data")
    
    def render_meal_logging_tab(self):
        """Render meal logging tab."""
        st.header("What did you eat?")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            log_date = st.date_input("Date", value=date.today())
            
            food_text = st.text_area(
                "Enter today's meals", 
                placeholder="e.g., 2 rotis, 1 cup dal, 1 glass milk, 200g chicken, 1 bowl rice",
                height=150,
                help="Separate multiple items with commas"
            )
            
            col_log, col_clear = st.columns(2)
            
            with col_log:
                if st.button("Log Meals", type="primary"):
                    if not food_text.strip():
                        st.error("Please enter some meals first!")
                    else:
                        try:
                            nutrition_info = parse_and_get_nutrition(food_text)
                            
                            total_logged = 0
                            for item in nutrition_info.get('found_items', []):
                                food_name = item.get('matched_name', 'Unknown')
                                quantity = item.get('quantity', 100)
                                cost = self.components['meal_economics'].calculate_meal_cost(food_name, quantity)
                                
                                entry = {
                                    "date": log_date,
                                    "food_name": food_name,
                                    "cuisine": "Auto-detected",
                                    "carbs_g": item.get('carbs_g', 0),
                                    "protein_g": item.get('protein_g', 0),
                                    "fat_g": item.get('fat_g', 0),
                                    "calories": item.get('calories', 0),
                                    "quantity": quantity,
                                    "unit": item.get('unit', 'g'),
                                    "original_input": item.get('original', ''),
                                    "matched_from_db": True,
                                    "cost": cost
                                }
                                indb_store.add_meal_entry(entry)
                                total_logged += 1
                            
                            st.success(f"Successfully logged {total_logged} food items!")
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error logging meals: {e}")
            
            with col_clear:
                if st.button("Clear Input"):
                    st.rerun()
        
        with col2:
            st.subheader("Quick Add")
            st.caption("Based on your recent meals:")
            
            frequent_meals = indb_store.get_most_frequent_meals(days=30, top_n=5)
            
            if frequent_meals:
                for display_name, food_input in frequent_meals:
                    if st.button(display_name, key=f"freq_{food_input}"):
                        try:
                            nutrition_info = parse_and_get_nutrition(food_input)
                            for item in nutrition_info.get('found_items', []):
                                food_name = item.get('matched_name', 'Unknown')
                                quantity = item.get('quantity', 100)
                                cost = self.components['meal_economics'].calculate_meal_cost(food_name, quantity)
                                
                                entry = {
                                    "date": log_date,
                                    "food_name": food_name,
                                    "cuisine": "Frequent",
                                    "carbs_g": item.get('carbs_g', 0),
                                    "protein_g": item.get('protein_g', 0),
                                    "fat_g": item.get('fat_g', 0),
                                    "calories": item.get('calories', 0),
                                    "quantity": quantity,
                                    "unit": item.get('unit', 'g'),
                                    "original_input": food_input,
                                    "matched_from_db": True,
                                    "cost": cost
                                }
                                indb_store.add_meal_entry(entry)
                            st.success(f"Added {display_name}")
                            st.rerun()
                        except Exception as e:
                            st.warning(f"Could not add {display_name}: {e}")
            else:
                # Default common meals if no history
                st.info("Log more meals to see your frequent choices")
                default_meals = [
                    ("2 Roti + Sabji", "2 roti, 1 bowl vegetable curry"),
                    ("Rice + Dal", "1 plate rice, 1 bowl dal"),
                    ("Chicken Curry", "200g chicken curry"),
                    ("Paneer Dish", "150g paneer sabji"),
                ]
                for display_name, food_input in default_meals:
                    if st.button(display_name, key=f"default_{food_input}"):
                        try:
                            nutrition_info = parse_and_get_nutrition(food_input)
                            for item in nutrition_info.get('found_items', []):
                                food_name = item.get('matched_name', 'Unknown')
                                quantity = item.get('quantity', 100)
                                cost = self.components['meal_economics'].calculate_meal_cost(food_name, quantity)
                                
                                entry = {
                                    "date": log_date,
                                    "food_name": food_name,
                                    "cuisine": "Quick Add",
                                    "carbs_g": item.get('carbs_g', 0),
                                    "protein_g": item.get('protein_g', 0),
                                    "fat_g": item.get('fat_g', 0),
                                    "calories": item.get('calories', 0),
                                    "quantity": quantity,
                                    "unit": item.get('unit', 'g'),
                                    "original_input": food_input,
                                    "matched_from_db": True,
                                    "cost": cost
                                }
                                indb_store.add_meal_entry(entry)
                            st.success(f"Added {display_name}")
                            st.rerun()
                        except Exception as e:
                            st.warning(f"Could not add {display_name}: {e}")
    
    def render_today_insights_tab(self):
        """Render today's insights tab."""
        st.header("Today's Meal Insights")
        
        # Get today meals
        today_meals = indb_store.get_meals_for_date(date.today())
        
        # Always show insights, even if no meals logged today
        if not today_meals.empty and len(today_meals) > 0:
            today_meals_list = today_meals.to_dict('records')
            
            # Create daily snapshot
            daily_snapshot = self.components['meal_aggregator'].create_daily_snapshot(today_meals_list, date.today())
            self.current_analysis = daily_snapshot
            
            # Get historical meals for ML insights
            historical_meals = indb_store.get_recent_meals(30)
            historical_meals_list = historical_meals.to_dict('records') if not historical_meals.empty else []
            
            # Main metrics row
            st.subheader("Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                cost_value = daily_snapshot.total_cost
                st.metric("Total Meal Cost", f"Rs {cost_value:.2f}")
            
            with col2:
                affordability_pct = daily_snapshot.affordability_score * 100
                st.metric("Affordability", f"{affordability_pct:.0f}%")
            
            with col3:
                cost_per_cal = daily_snapshot.cost_per_calorie
                st.metric("Cost per Calorie", f"Rs {cost_per_cal:.4f}")
            
            with col4:
                st.metric("Total Calories", f"{daily_snapshot.total_calories:.0f}")
            
            # Nutrition metrics
            st.subheader("Nutrition Breakdown")
            col_p, col_c, col_f, col_b = st.columns(4)
            
            with col_p:
                protein_pct = (daily_snapshot.total_protein / 50) * 100
                st.metric("Protein", f"{daily_snapshot.total_protein:.1f}g", f"{protein_pct:.0f}% RDA")
            
            with col_c:
                st.metric("Carbohydrates", f"{daily_snapshot.total_carbs:.1f}g")
            
            with col_f:
                st.metric("Fat", f"{daily_snapshot.total_fat:.1f}g")
            
            with col_b:
                budget_used = (daily_snapshot.total_cost / 500) * 100
                st.metric("Budget Used", f"{budget_used:.0f}%")
            
            # Cost breakdown chart
            if hasattr(daily_snapshot, 'category_costs') and daily_snapshot.category_costs:
                st.subheader("Cost Distribution by Category")
                
                fig_pie = px.pie(
                    values=list(daily_snapshot.category_costs.values()),
                    names=list(daily_snapshot.category_costs.keys()),
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, config={'displayModeBar': False}, width='stretch')
            
            # Enhanced Cost insights with ML
            st.subheader("Smart Insights")
            insights = self.components['meal_economics'].generate_cost_insights(daily_snapshot, historical_meals_list)
            
            for insight in insights:
                if "Cost Alert" in insight or "High cost alert" in insight or "alert" in insight.lower():
                    st.error(insight)
                elif "Excellent" in insight.lower() or "efficient" in insight.lower():
                    st.success(insight)
                elif "consider" in insight.lower() or "High cost" in insight:
                    st.warning(insight)
                else:
                    st.info(insight)
            
            # Show meal items
            st.subheader("Today's Meals")
            meal_items = getattr(daily_snapshot, 'meal_items', [])
            if meal_items:
                display_data = []
                for item in meal_items:
                    display_data.append({
                        'Food': item.get('food_name', 'Unknown'),
                        'Quantity': f"{item.get('quantity', 0)}g",
                        'Calories': item.get('calories', 0),
                        'Protein (g)': item.get('protein_g', 0),
                        'Cost (Rs)': f"Rs {item.get('cost', 0):.2f}"
                    })
                
                display_df = pd.DataFrame(display_data)
                st.dataframe(display_df, width='stretch', hide_index=True)
        
        else:
            # Show sample insights when no meals logged today
            st.info("No meals logged today. Sample insights based on historical data:")
            
            # Create sample insights
            st.subheader("Sample Insights")
            st.info("Based on your eating patterns, you typically spend Rs 180-250 per day")
            st.warning("Consider adding more protein-rich foods to your diet")
            st.success("Your cost efficiency is good at Rs 0.006 per calorie")
            
            st.subheader("Quick Start")
            st.write("1. Use the 'Log Meals' tab to add today's meals")
            st.write("2. Check the 'Trends' tab to see your spending patterns")
            st.write("3. Visit 'Cost Recommendations' for money-saving tips")
    
    def render_trends_tab(self):
        """Render trends tab."""
        st.header("Cost & Nutrition Trends")
        
        # Get recent meals (sample data ensures we always have data)
        recent_meals = indb_store.get_recent_meals(30)
        
        if not recent_meals.empty and len(recent_meals) > 0:
            recent_meals_list = recent_meals.to_dict('records')
            
            days = st.selectbox("Analysis Period", [7, 14, 30], index=2, key="trend_days")
            
            cost_analysis = self.components['meal_aggregator'].get_cost_analysis(recent_meals_list, days)
            
            if cost_analysis.get('daily_snapshots'):
                daily_data = cost_analysis['daily_snapshots']
                df_costs = pd.DataFrame(daily_data)
                
                if 'date' in df_costs.columns and not df_costs.empty:
                    df_costs['date'] = pd.to_datetime(df_costs['date'])
                    
                    st.subheader("Cost Trends")
                    fig_costs = go.Figure()
                    fig_costs.add_trace(go.Scatter(
                        x=df_costs['date'],
                        y=df_costs['total_cost'],
                        mode='lines+markers',
                        name='Daily Cost',
                        line=dict(color='#FF6B6B', width=3)
                    ))
                    
                    # Add budget line
                    fig_costs.add_hline(y=500, line_dash="dash", line_color="red", 
                                      annotation_text="Daily Budget Limit")
                    
                    fig_costs.update_layout(
                        height=400,
                        title="Daily Meal Costs Over Time",
                        xaxis_title="Date",
                        yaxis_title="Cost (Rs)"
                    )
                    st.plotly_chart(fig_costs, config={'displayModeBar': False}, width='stretch')
                    
                    st.subheader("Nutrition Trends")
                    fig_nutrition = make_subplots(
                        rows=2, cols=2, 
                        subplot_titles=('Calories', 'Protein (g)', 'Carbs (g)', 'Fat (g)')
                    )
                    
                    fig_nutrition.add_trace(
                        go.Scatter(x=df_costs['date'], y=df_costs['total_calories'], 
                                 name='Calories', line=dict(color='#4ECDC4')),
                        row=1, col=1
                    )
                    fig_nutrition.add_trace(
                        go.Scatter(x=df_costs['date'], y=df_costs['total_protein'], 
                                 name='Protein', line=dict(color='#45B7D1')),
                        row=1, col=2
                    )
                    fig_nutrition.add_trace(
                        go.Scatter(x=df_costs['date'], y=df_costs['total_carbs'], 
                                 name='Carbs', line=dict(color='#96CEB4')),
                        row=2, col=1
                    )
                    fig_nutrition.add_trace(
                        go.Scatter(x=df_costs['date'], y=df_costs['total_fat'], 
                                 name='Fat', line=dict(color='#FFEAA7')),
                        row=2, col=2
                    )
                    
                    fig_nutrition.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig_nutrition, config={'displayModeBar': False}, width='stretch')
        
        else:
            st.info("Sample data loaded. Start logging meals to see your personal trends!")

# Main execution
if __name__ == "__main__":
    # Clear any cached modules to ensure fresh import
    if 'proj_ui' in sys.modules:
        del sys.modules['proj_ui']
    
    try:
        ui = MealCostOptimizerUI()
        ui.render()
    except Exception as e:
        st.error(f"Failed to load application: {e}")