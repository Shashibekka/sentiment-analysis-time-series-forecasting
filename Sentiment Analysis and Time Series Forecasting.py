# Complete Real Data Analysis System
# =============================================================================
# SENTIMENT ANALYSIS AND TIME SERIES FORECASTING WITH REAL DATA
# Uses: Twitter_train.csv, twitter_validation.csv, Google_Stock_Price_Train.csv
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP and ML Libraries
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import json
import os
from datetime import datetime

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    print("Warning: Could not download NLTK data")

class ComprehensiveFinancialAnalyzer:
    """
    Complete financial analysis system using real Twitter and stock data
    """
    
    def __init__(self):
        self.twitter_train = None
        self.twitter_validation = None
        self.stock_data = None
        self.trained_models = {}
        self.model_performance = {}
        self.forecasting_results = {}
        self.web_data = {}
        
        # Initialize sentiment analyzer
        try:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        except:
            self.vader_analyzer = None
            print("Warning: VADER sentiment analyzer not available")
    
    def load_datasets(self):
        """Load all required datasets"""
        print("📊 Loading Real Datasets...")
        print("="*50)
        
        # Load Twitter validation data
        try:
            self.twitter_validation = pd.read_csv('twitter_validation.csv')
            self.twitter_validation.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
            
            # Filter valid sentiments
            valid_sentiments = ['Positive', 'Negative', 'Neutral']
            self.twitter_validation = self.twitter_validation[
                self.twitter_validation['Sentiment'].isin(valid_sentiments)
            ]
            
            print(f"✅ Twitter validation data: {len(self.twitter_validation)} tweets")
            print(f"   Sentiment distribution: {dict(self.twitter_validation['Sentiment'].value_counts())}")
            
        except Exception as e:
            print(f"❌ Error loading Twitter validation data: {e}")
            return False
        
        # Load Twitter training data (or use validation if not available)
        try:
            self.twitter_train = pd.read_csv('twitter_training.csv')
            if len(self.twitter_train.columns) == 4:
                self.twitter_train.columns = ['ID', 'Company', 'Sentiment', 'Tweet']
            
            # Filter valid sentiments
            self.twitter_train = self.twitter_train[
                self.twitter_train['Sentiment'].isin(valid_sentiments)
            ]
            
            print(f"✅ Twitter training data: {len(self.twitter_train)} tweets")
            
        except FileNotFoundError:
            print("ℹ️  Twitter_train.csv not found, using validation data split for training")
            # Split validation data for training
            train_data = self.twitter_validation.sample(frac=0.7, random_state=42)
            self.twitter_train = train_data
            self.twitter_validation = self.twitter_validation.drop(train_data.index)
            print(f"✅ Created training split: {len(self.twitter_train)} tweets")
        
        # Load Google stock data
        try:
            self.stock_data = pd.read_csv('Google_Stock_Price_Train.csv')
            
            # Clean and convert data types
            for col in ['Open', 'High', 'Low', 'Close']:
                self.stock_data[col] = self.stock_data[col].astype(str).str.replace(',', '').astype(float)
            
            self.stock_data['Volume'] = self.stock_data['Volume'].astype(str).str.replace(',', '').str.replace('"', '').astype(int)
            self.stock_data['Date'] = pd.to_datetime(self.stock_data['Date'])
            self.stock_data = self.stock_data.sort_values('Date').reset_index(drop=True)
            
            print(f"✅ Google stock data: {len(self.stock_data)} records")
            print(f"   Date range: {self.stock_data['Date'].min()} to {self.stock_data['Date'].max()}")
            print(f"   Price range: ${self.stock_data['Close'].min():.2f} - ${self.stock_data['Close'].max():.2f}")
            
        except Exception as e:
            print(f"❌ Error loading stock data: {e}")
            return False
        
        return True
    
    def preprocess_text(self, text):
        """Advanced text preprocessing for financial sentiment analysis"""
        if pd.isna(text):
            return ""
        
        import re
        text = str(text).lower()
        
        # Remove URLs, mentions, hashtags (keep content)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but preserve financial symbols
        text = re.sub(r'[^\w\s$%]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def train_sentiment_models(self):
        """Train multiple sentiment analysis models on real Twitter data"""
        print("\n🤖 Training Sentiment Analysis Models...")
        print("-"*50)
        
        # Preprocess training data
        self.twitter_train['processed_tweet'] = self.twitter_train['Tweet'].apply(self.preprocess_text)
        self.twitter_train = self.twitter_train[self.twitter_train['processed_tweet'].str.len() > 0]
        
        X = self.twitter_train['processed_tweet']
        y = self.twitter_train['Sentiment']
        
        # Split for model validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} tweets")
        print(f"Test set: {len(X_test)} tweets")
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2, max_df=0.95)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train multiple models
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        }
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            
            # Train model
            model.fit(X_train_vec, y_train)
            
            # Store model and vectorizer
            self.trained_models[name] = {
                'model': model,
                'vectorizer': vectorizer
            }
            
            # Evaluate on test set
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            self.model_performance[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'training_size': len(X_train),
                'test_size': len(X_test)
            }
            
            print(f"   ✅ {name}: {accuracy:.3f} accuracy")
        
        # Find best model
        best_model = max(self.model_performance, key=lambda x: self.model_performance[x]['accuracy'])
        print(f"\n🏆 Best Model: {best_model} ({self.model_performance[best_model]['accuracy']:.3f} accuracy)")
        
        return best_model
    
    def validate_sentiment_models(self):
        """Validate models on real validation dataset"""
        print("\n📊 Validating Models on Real Data...")
        print("-"*40)
        
        # Preprocess validation data
        self.twitter_validation['processed_tweet'] = self.twitter_validation['Tweet'].apply(self.preprocess_text)
        self.twitter_validation = self.twitter_validation[self.twitter_validation['processed_tweet'].str.len() > 0]
        
        validation_results = {}
        
        for name, model_info in self.trained_models.items():
            model = model_info['model']
            vectorizer = model_info['vectorizer']
            
            # Transform validation data
            X_val = vectorizer.transform(self.twitter_validation['processed_tweet'])
            y_val = self.twitter_validation['Sentiment']
            
            # Predict
            predictions = model.predict(X_val)
            accuracy = accuracy_score(y_val, predictions)
            
            validation_results[name] = {
                'validation_accuracy': accuracy,
                'predictions': predictions
            }
            
            print(f"{name}: {accuracy:.3f} validation accuracy")
        
        return validation_results
    
    def analyze_sample_tweets(self):
        """Comprehensive analysis of sample tweets"""
        print("\n🔍 Analyzing Sample Tweets...")
        print("-"*30)
        
        sample_analyses = []
        
        # Get best model
        best_model_name = max(self.model_performance, key=lambda x: self.model_performance[x]['accuracy'])
        best_model_info = self.trained_models[best_model_name]
        
        # Analyze first 10 validation tweets
        for idx, row in self.twitter_validation.head(10).iterrows():
            text = self.preprocess_text(row['Tweet'])
            
            # TextBlob analysis
            blob = TextBlob(text)
            tb_polarity = blob.sentiment.polarity
            tb_sentiment = 'Positive' if tb_polarity > 0.1 else 'Negative' if tb_polarity < -0.1 else 'Neutral'
            
            # VADER analysis
            if self.vader_analyzer:
                vader_scores = self.vader_analyzer.polarity_scores(text)
                vader_compound = vader_scores['compound']
                vader_sentiment = 'Positive' if vader_compound >= 0.05 else 'Negative' if vader_compound <= -0.05 else 'Neutral'
            else:
                vader_sentiment = 'N/A'
                vader_compound = 0
            
            # ML prediction
            X_vec = best_model_info['vectorizer'].transform([text])
            ml_pred = best_model_info['model'].predict(X_vec)[0]
            ml_prob = max(best_model_info['model'].predict_proba(X_vec)[0])
            
            analysis = {
                'tweet': row['Tweet'][:100] + "..." if len(row['Tweet']) > 100 else row['Tweet'],
                'actual': row['Sentiment'],
                'company': row['Company'],
                'textblob': {'prediction': tb_sentiment, 'score': round(tb_polarity, 3)},
                'vader': {'prediction': vader_sentiment, 'score': round(vader_compound, 3)},
                'ml': {'prediction': ml_pred, 'confidence': round(ml_prob, 3)}
            }
            
            sample_analyses.append(analysis)
            
            if idx < 3:  # Display first 3
                print(f"\nSample {idx + 1}:")
                print(f"Tweet: {analysis['tweet']}")
                print(f"Actual: {analysis['actual']} | Company: {analysis['company']}")
                print(f"TextBlob: {analysis['textblob']['prediction']} ({analysis['textblob']['score']})")
                print(f"VADER: {analysis['vader']['prediction']} ({analysis['vader']['score']})")
                print(f"ML: {analysis['ml']['prediction']} ({analysis['ml']['confidence']})")
        
        return sample_analyses
    
    def train_time_series_models(self):
        """Train ARIMA and SARIMA models on Google stock data with train-test-validation split"""
        print("\n📈 Training Time Series Models...")
        print("-"*40)
        
        prices = self.stock_data['Close'].values
        
        # Split data into train (70%), validation (15%), and test (15%)
        train_size = int(len(prices) * 0.7)
        val_size = int(len(prices) * 0.15)
        
        train_data = prices[:train_size]
        val_data = prices[train_size:train_size+val_size]
        test_data = prices[train_size+val_size:]
        
        print(f"   📊 Data Split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
        
        # Train and validate ARIMA model
        try:
            print("🔄 Training ARIMA model...")
            arima_order = (2, 1, 2)
            arima_model = ARIMA(train_data, order=arima_order)
            arima_fitted = arima_model.fit()
            
            # Validate on validation data
            arima_val_forecast = arima_fitted.forecast(steps=len(val_data))
            arima_val_mse = np.mean((arima_val_forecast - val_data)**2)
            
            print(f"   📊 ARIMA Validation MSE: {arima_val_mse:.2f}")
            
            # Retrain on train + validation data for final model
            arima_train_val_data = np.concatenate([train_data, val_data])
            arima_model_final = ARIMA(arima_train_val_data, order=arima_order)
            arima_fitted_final = arima_model_final.fit()
            
            # Forecast on test data length and future
            arima_test_forecast = arima_fitted_final.forecast(steps=len(test_data))
            arima_test_mse = np.mean((arima_test_forecast - test_data)**2)
            
            # Forecast future 30 days
            arima_future_forecast = arima_fitted_final.forecast(steps=30)
            
            self.forecasting_results['ARIMA'] = {
                'model': arima_fitted_final,
                'forecast': arima_future_forecast.tolist(),
                'aic': float(arima_fitted_final.aic),
                'bic': float(arima_fitted_final.bic),
                'order': str(arima_order),
                'val_mse': float(arima_val_mse),
                'test_mse': float(arima_test_mse),
                'next_5_days': arima_future_forecast[:5].tolist()
            }
            
            print(f"   ✅ ARIMA trained (AIC: {arima_fitted_final.aic:.2f})")
            print(f"   📊 Test MSE: {arima_test_mse:.2f}")
            print(f"   📊 Next 5 days: {[f'${x:.2f}' for x in arima_future_forecast[:5]]}")
            
        except Exception as e:
            print(f"   ❌ ARIMA training failed: {str(e)[:50]}...")
            self.forecasting_results['ARIMA'] = {'error': str(e)}
        
        # Train and validate SARIMA model
        try:
            print("🔄 Training SARIMA model...")
            sarima_order = (1, 1, 1)
            sarima_seasonal_order = (1, 1, 1, 12)
            sarima_model = SARIMAX(train_data, order=sarima_order, seasonal_order=sarima_seasonal_order)
            sarima_fitted = sarima_model.fit(disp=False)
            
            # Validate on validation data
            sarima_val_forecast = sarima_fitted.forecast(steps=len(val_data))
            sarima_val_mse = np.mean((sarima_val_forecast - val_data)**2)
            
            print(f"   📊 SARIMA Validation MSE: {sarima_val_mse:.2f}")
            
            # Retrain on train + validation data for final model
            sarima_train_val_data = np.concatenate([train_data, val_data])
            sarima_model_final = SARIMAX(sarima_train_val_data, order=sarima_order, seasonal_order=sarima_seasonal_order)
            sarima_fitted_final = sarima_model_final.fit(disp=False)
            
            # Forecast on test data length and future
            sarima_test_forecast = sarima_fitted_final.forecast(steps=len(test_data))
            sarima_test_mse = np.mean((sarima_test_forecast - test_data)**2)
            
            # Forecast future 30 days
            sarima_future_forecast = sarima_fitted_final.forecast(steps=30)
            
            self.forecasting_results['SARIMA'] = {
                'model': sarima_fitted_final,
                'forecast': sarima_future_forecast.tolist(),
                'aic': float(sarima_fitted_final.aic),
                'bic': float(sarima_fitted_final.bic),
                'order': f'{sarima_order}x{sarima_seasonal_order}',
                'val_mse': float(sarima_val_mse),
                'test_mse': float(sarima_test_mse),
                'next_5_days': sarima_future_forecast[:5].tolist()
            }
            
            print(f"   ✅ SARIMA trained (AIC: {sarima_fitted_final.aic:.2f})")
            print(f"   📊 Test MSE: {sarima_test_mse:.2f}")
            print(f"   📊 Next 5 days: {[f'${x:.2f}' for x in sarima_future_forecast[:5]]}")
            
        except Exception as e:
            print(f"   ❌ SARIMA training failed: {str(e)[:50]}...")
            self.forecasting_results['SARIMA'] = {'error': str(e)}
        
        return self.forecasting_results

    
    def generate_web_data(self, sample_analyses):
        """Generate data for web application"""
        print("\n🌐 Generating Web Application Data...")
        print("-"*35)
        
        # Best model info
        best_model_name = max(self.model_performance, key=lambda x: self.model_performance[x]['accuracy'])
        best_accuracy = self.model_performance[best_model_name]['accuracy']
        
        # Prepare web data
        self.web_data = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'datasets_used': {
                    'twitter_train_size': len(self.twitter_train),
                    'twitter_validation_size': len(self.twitter_validation),
                    'stock_data_size': len(self.stock_data)
                }
            },
            'sentiment_analysis': {
                'total_tweets': len(self.twitter_validation),
                'sentiment_distribution': dict(self.twitter_validation['Sentiment'].value_counts()),
                'company_distribution': dict(self.twitter_validation['Company'].value_counts().head(10)),
                'model_performance': {k: {'accuracy': v['accuracy']} for k, v in self.model_performance.items()},
                'best_model': {
                    'name': best_model_name,
                    'accuracy': best_accuracy
                },
                'sample_analyses': sample_analyses[:10]
            },
            'stock_forecasting': {
                'stock_stats': {
                    'min_price': float(self.stock_data['Close'].min()),
                    'max_price': float(self.stock_data['Close'].max()),
                    'avg_price': float(self.stock_data['Close'].mean()),
                    'latest_price': float(self.stock_data['Close'].iloc[-1]),
                    'total_records': len(self.stock_data)
                },
                'forecasting_results': {}
            }
        }
        
        # Add forecasting results
        for model_name, results in self.forecasting_results.items():
            if 'forecast' in results:  # Only successful models
                self.web_data['stock_forecasting']['forecasting_results'][model_name] = {
                    'next_5_days': results['next_5_days'],
                    'full_forecast': results['forecast'],
                    'aic': results.get('aic', 0),
                    'order': results.get('order', '')
                }
        
        # Save web data to JSON
        with open('web_app_data.json', 'w') as f:
            json.dump(self.web_data, f, indent=2, default=str)
        
        print("   ✅ Web data generated and saved to 'web_app_data.json'")
        return self.web_data
    
    def run_complete_analysis(self):
        """Execute the complete analysis pipeline"""
        print("🚀 STARTING COMPREHENSIVE REAL DATA ANALYSIS")
        print("="*80)
        
        # Load datasets
        if not self.load_datasets():
            print("❌ Failed to load datasets. Please check file paths.")
            return False
        
        # Train sentiment models
        best_model = self.train_sentiment_models()
        
        # Validate models
        validation_results = self.validate_sentiment_models()
        
        # Analyze sample tweets
        sample_analyses = self.analyze_sample_tweets()
        
        # Train time series models
        forecasting_results = self.train_time_series_models()
        
        # Generate web application data
        web_data = self.generate_web_data(sample_analyses)
        
        # Print final summary
        self.print_final_summary()
        
        return True
    
    def print_final_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("🎯 ANALYSIS COMPLETE - SUMMARY REPORT")
        print("="*80)
        
        # Dataset summary
        print(f"\n📊 DATASETS PROCESSED:")
        print(f"   • Twitter Training: {len(self.twitter_train):,} tweets")
        print(f"   • Twitter Validation: {len(self.twitter_validation):,} tweets")
        print(f"   • Google Stock Data: {len(self.stock_data):,} records")
        
        # Model performance
        print(f"\n🤖 SENTIMENT ANALYSIS RESULTS:")
        best_model_name = max(self.model_performance, key=lambda x: self.model_performance[x]['accuracy'])
        best_accuracy = self.model_performance[best_model_name]['accuracy']
        print(f"   🏆 Best Model: {best_model_name}")
        print(f"   📈 Best Accuracy: {best_accuracy:.1%}")
        print(f"   📊 Models Trained: {len(self.model_performance)}")
        
        # Forecasting results
        successful_forecasts = [name for name, result in self.forecasting_results.items() if 'forecast' in result]
        print(f"\n📈 TIME SERIES FORECASTING:")
        print(f"   ✅ Successful Models: {', '.join(successful_forecasts)}")
        
        for model_name in successful_forecasts:
            results = self.forecasting_results[model_name]
            avg_forecast = np.mean(results['next_5_days'])
            print(f"   📊 {model_name} 5-day avg: ${avg_forecast:.2f}")
        
        # Files generated
        print(f"\n📁 FILES GENERATED:")
        print(f"   • web_app_data.json (for web application)")
        print(f"   • Analysis ready for HTML/CSS/JS integration")
        
        print(f"\n✅ Complete analysis pipeline executed successfully!")

def main():
    """Main execution function"""
    analyzer = ComprehensiveFinancialAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 Ready for web application integration!")
        return analyzer
    else:
        print("\n❌ Analysis failed. Please check data files.")
        return None

if __name__ == "__main__":
    analyzer = main()
