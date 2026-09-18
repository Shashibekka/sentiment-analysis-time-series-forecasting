# Financial Sentiment Analysis & Stock Forecasting Dashboard

This project implements a comprehensive financial analysis system that combines sentiment analysis of Twitter data with stock price forecasting using time series models. It features a professional web interface to visualize the results.

## Project Overview

The system processes real Twitter data for sentiment analysis and Google stock price data for forecasting. Sentiment analysis is performed using multiple machine learning models (Naive Bayes, Logistic Regression, Random Forest), alongside TextBlob and VADER for comparison. Stock price forecasting is conducted using ARIMA and SARIMA models with a train-test-validation split for improved accuracy.

## Features

- **Sentiment Analysis**: Analyzes Twitter data to classify sentiments as Positive, Negative, or Neutral.
- **Stock Price Forecasting**: Uses ARIMA and SARIMA models to predict future stock prices for Google.
- **Interactive Dashboard**: A web interface built with HTML, CSS, and JavaScript to display analysis results and forecasts.
- **Real Data**: Utilizes `twitter_training.csv`, `twitter_validation.csv`, and `Google_Stock_Price_Train.csv` datasets.

## Project Structure

```
FinancialDashboard/
│
├── Frontend_web/
│   ├── index.html          # Main HTML file for the dashboard
│   ├── style.css           # CSS styling for the dashboard
│   └── app.js              # JavaScript for dashboard functionality
│
├── data/
│   ├── twitter_training.csv      # Twitter training dataset for sentiment analysis
│   ├── twitter_validation.csv    # Twitter validation dataset for sentiment analysis
│   └── Google_Stock_Price_Train.csv  # Google stock price dataset for forecasting
│
├── Sentiment-Analysis-and-Time-Series-Forecasting.py  # Main Python script for analysis
├── web_app_data.json         # Generated JSON data for web interface
├── requirements.txt          # List of Python dependencies
└── README.md                 # Project documentation
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Git (for version control)
- A web browser for viewing the dashboard

### Steps
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd FinancialDashboard
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Analysis Script**:
   ```bash
   python Sentiment-Analysis-and-Time-Series-Forecasting.py
   ```
   This will process the data, train models, and generate `web_app_data.json`.

5. **View the Dashboard**:
   - Open `Frontend_web/index.html` in a web browser to see the results.
   - Alternatively, use a local server for better performance (e.g., with Python):
     ```bash
     python -m http.server 8000
     ```
     Then navigate to `http://localhost:8000/Frontend_web/index.html`.

## Usage

- **Data Analysis**: The Python script processes datasets located in the `data` folder and outputs results to `web_app_data.json`.
- **Dashboard**: The frontend web interface visualizes sentiment distributions, model performance, stock forecasts, and sample tweet analyses.
- **Customization**: Modify model parameters in `Sentiment-Analysis-and-Time-Series-Forecasting.py` or update the UI in the `Frontend_web` files as needed.

## Model Performance

The system achieves the following performance metrics:
- **Sentiment Analysis**: Up to 60.2% accuracy using Naive Bayes classifier
- **Time Series Forecasting**: ARIMA and SARIMA models with train-test-validation split for improved accuracy
- **Real-time Processing**: Handles 828+ tweets and 1,258+ stock records

## Datasets

- **Twitter Training Data** (`twitter_training.csv`): Used to train sentiment analysis models.
- **Twitter Validation Data** (`twitter_validation.csv`): Used to validate sentiment analysis models.
- **Google Stock Price Data** (`Google_Stock_Price_Train.csv`): Historical stock prices for training ARIMA and SARIMA models.

## Technical Implementation

### Sentiment Analysis Models
- **Naive Bayes**: Primary classifier with highest accuracy
- **Logistic Regression**: Secondary classifier for comparison
- **Random Forest**: Ensemble method for robust predictions
- **TextBlob & VADER**: Rule-based sentiment analysis for benchmarking

### Time Series Models
- **ARIMA**: AutoRegressive Integrated Moving Average for trend analysis
- **SARIMA**: Seasonal ARIMA for capturing seasonal patterns
- **Train-Test-Validation Split**: 70%-15%-15% split for robust evaluation

### Web Interface
- **Responsive Design**: Professional dashboard with interactive charts
- **Real-time Updates**: Dynamic data visualization
- **Chart.js Integration**: Advanced charting capabilities

## Dependencies

See `requirements.txt` for a full list of Python packages required for this project.

Key dependencies include:
- pandas, numpy for data manipulation
- scikit-learn for machine learning models
- statsmodels for time series analysis
- nltk, textblob, vaderSentiment for text processing
- matplotlib, seaborn for visualization

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues
- **Model Training Errors**: Ensure all datasets are in the correct format and location
- **Dashboard Not Loading**: Check that `web_app_data.json` is generated properly
- **Package Installation Issues**: Use a virtual environment and update pip

### Support
For questions or issues, please:
1. Check the existing issues in the repository
2. Create a new issue with detailed description
3. Contact the project maintainer

## Future Enhancements

- Integration with real-time Twitter API
- Additional machine learning models
- Enhanced forecasting algorithms
- Mobile-responsive improvements
- Database integration for persistent storage

## Acknowledgments

- Twitter API for data access
- Google Finance for stock data
- Open source libraries and contributors
- Community feedback and suggestions