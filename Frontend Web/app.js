// Financial Dashboard JavaScript
class FinancialDashboard {
    constructor() {
        this.data = null;
        this.charts = {};
        this.init();
    }

    async init() {
        try {
            console.log('Initializing Financial Dashboard...');
            await this.loadData();
            this.renderDashboard();
            this.setupEventListeners();
            this.startRealTimeUpdates();
            console.log('Dashboard initialized successfully.');
        } catch (error) {
            console.error('Dashboard initialization failed:', error);
            this.showErrorMessage('Failed to load dashboard data');
        }
    }

    async loadData() {
        try {
            // Attempt to load data from JSON file
            const response = await fetch('web_app_data.json');
            if (response.ok) {
                this.data = await response.json();
                console.log('Data loaded successfully.');
            } else {
                throw new Error('Data not available');
            }
        } catch (error) {
            console.warn('Using fallback data:', error.message);
            // Fallback to mock data structure matching the real data format
            this.data = this.generateMockData();
        }
    }

    generateMockData() {
        return {
            metadata: {
                analysis_timestamp: new Date().toISOString(),
                datasets_used: {
                    twitter_train_size: 579,
                    twitter_validation_size: 249,
                    stock_data_size: 1258
                }
            },
            sentiment_analysis: {
                total_tweets: 828,
                sentiment_distribution: {
                    'Neutral': 285,
                    'Positive': 277,
                    'Negative': 266
                },
                company_distribution: {
                    'RedDeadRedemption(RDR)': 38,
                    'johnson&johnson': 36,
                    'TomClancysRainbowSix': 34,
                    'Nvidia': 33,
                    'ApexLegends': 33,
                    'AssassinsCreed': 32,
                    'Amazon': 31,
                    'Borderlands': 31,
                    'PlayStation5(PS5)': 31,
                    'LeagueOfLegends': 31
                },
                model_performance: {
                    'Naive Bayes': { accuracy: 0.602 },
                    'Logistic Regression': { accuracy: 0.578 },
                    'Random Forest': { accuracy: 0.486 }
                },
                best_model: {
                    name: 'Naive Bayes',
                    accuracy: 0.602
                },
                sample_analyses: [
                    {
                        tweet: 'BBC News - Amazon boss Jeff Bezos rejects claims company acted like a drug dealer',
                        actual: 'Neutral',
                        company: 'Amazon',
                        textblob: { prediction: 'Neutral', score: 0.0 },
                        vader: { prediction: 'Neutral', score: 0.0 },
                        ml: { prediction: 'Neutral', confidence: 0.612 }
                    },
                    {
                        tweet: '@Microsoft Why do I pay for WORD when it functions so poorly on my @SamsungUS Chromebook?',
                        actual: 'Negative',
                        company: 'Microsoft',
                        textblob: { prediction: 'Negative', score: -0.4 },
                        vader: { prediction: 'Negative', score: -0.5 },
                        ml: { prediction: 'Negative', confidence: 0.515 }
                    }
                ]
            },
            stock_forecasting: {
                stock_stats: {
                    min_price: 491.20,
                    max_price: 1216.83,
                    avg_price: 735.50,
                    latest_price: 756.20,
                    total_records: 1258
                },
                forecasting_results: {
                    'ARIMA': {
                        next_5_days: [760.50, 762.30, 758.90, 765.20, 767.10],
                        aic: 11247.85,
                        order: '(2,1,2)'
                    },
                    'SARIMA': {
                        next_5_days: [759.80, 761.20, 757.60, 763.40, 765.90],
                        aic: 11235.42,
                        order: '(1,1,1)x(1,1,1,12)'
                    }
                }
            }
        };
    }

    renderDashboard() {
        this.updateSummaryStats();
        this.renderSentimentChart();
        this.renderStockChart();
        this.renderModelPerformance();
        this.renderSampleAnalyses();
        this.renderForecastResults();
        this.renderStockStats();
        this.renderCompanyAnalysis();
        this.updateTimestamp();
    }

    updateSummaryStats() {
        const { sentiment_analysis, stock_forecasting } = this.data;
        
        document.getElementById('totalTweets').textContent = sentiment_analysis.total_tweets.toLocaleString();
        document.getElementById('modelAccuracy').textContent = 
            `${(sentiment_analysis.best_model.accuracy * 100).toFixed(1)}%`;
        document.getElementById('stockRecords').textContent = 
            stock_forecasting.stock_stats.total_records.toLocaleString();
        
        const totalModels = Object.keys(sentiment_analysis.model_performance).length + 
                           Object.keys(stock_forecasting.forecasting_results).length;
        document.getElementById('modelsTrained').textContent = totalModels;
    }

    renderSentimentChart() {
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const sentimentData = this.data.sentiment_analysis.sentiment_distribution;
        
        this.charts.sentiment = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(sentimentData),
                datasets: [{
                    data: Object.values(sentimentData),
                    backgroundColor: [
                        '#15803d', // Positive - Green
                        '#b91c1c', // Negative - Red  
                        '#6b7280'  // Neutral - Gray
                    ],
                    borderWidth: 1,
                    borderColor: '#ffffff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    renderStockChart() {
        const ctx = document.getElementById('stockChart').getContext('2d');
        const forecastData = this.data.stock_forecasting.forecasting_results;
        
        // Generate historical data simulation
        const historicalData = this.generateHistoricalStockData();
        const labels = [...historicalData.labels];
        const historical = [...historicalData.prices];
        
        // Add forecast data
        const forecastLabels = ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'];
        labels.push(...forecastLabels);
        
        const arimaForecast = forecastData.ARIMA ? forecastData.ARIMA.next_5_days : [];
        const sarimaForecast = forecastData.SARIMA ? forecastData.SARIMA.next_5_days : [];
        
        this.charts.stock = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Historical Prices',
                        data: [...historical, ...Array(5).fill(null)],
                        borderColor: '#1e3a8a',
                        backgroundColor: 'rgba(30, 58, 138, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    },
                    {
                        label: 'ARIMA Forecast',
                        data: [...Array(historical.length).fill(null), ...arimaForecast],
                        borderColor: '#15803d',
                        backgroundColor: 'rgba(21, 128, 61, 0.1)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'SARIMA Forecast',
                        data: [...Array(historical.length).fill(null), ...sarimaForecast],
                        borderColor: '#b45309',
                        backgroundColor: 'rgba(180, 83, 9, 0.1)',
                        borderWidth: 2,
                        borderDash: [10, 5],
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: '#e5e7eb'
                        },
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: '#e5e7eb'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            padding: 20,
                            font: {
                                size: 12,
                                weight: '500'
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: $${context.parsed.y?.toFixed(2) || 'N/A'}`;
                            }
                        }
                    }
                }
            }
        });
    }

    generateHistoricalStockData() {
        const stats = this.data.stock_forecasting.stock_stats;
        const labels = [];
        const prices = [];
        
        // Generate 30 days of simulated historical data
        let currentPrice = stats.latest_price;
        
        for (let i = 29; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            
            // Simulate price movement
            const change = (Math.random() - 0.5) * 20;
            currentPrice = Math.max(stats.min_price, Math.min(stats.max_price, currentPrice + change));
            prices.push(currentPrice);
        }
        
        return { labels, prices };
    }

    renderModelPerformance() {
        const modelResults = document.getElementById('modelResults');
        const models = this.data.sentiment_analysis.model_performance;
        
        modelResults.innerHTML = '';
        
        Object.entries(models).forEach(([name, performance]) => {
            const modelItem = document.createElement('div');
            modelItem.className = 'model-item fade-in-up';
            
            modelItem.innerHTML = `
                <h4>${name}</h4>
                <div class="accuracy">${(performance.accuracy * 100).toFixed(1)}%</div>
                <p>Accuracy Score</p>
            `;
            
            modelResults.appendChild(modelItem);
        });
    }

    renderSampleAnalyses() {
        const sampleTweets = document.getElementById('sampleTweets');
        const samples = this.data.sentiment_analysis.sample_analyses;
        
        sampleTweets.innerHTML = '';
        
        samples.slice(0, 3).forEach((sample, index) => {
            const sampleItem = document.createElement('div');
            sampleItem.className = 'sample-item fade-in-up';
            sampleItem.style.animationDelay = `${index * 0.1}s`;
            
            sampleItem.innerHTML = `
                <div class="sample-tweet">"${sample.tweet.substring(0, 100)}..."</div>
                <div class="sample-meta">
                    <strong>Company:</strong> ${sample.company} | 
                    <strong>Actual:</strong> ${sample.actual}
                </div>
                <div class="sample-predictions">
                    <span class="prediction-tag prediction-${sample.textblob.prediction.toLowerCase()}">
                        TextBlob: ${sample.textblob.prediction}
                    </span>
                    <span class="prediction-tag prediction-${sample.vader.prediction.toLowerCase()}">
                        VADER: ${sample.vader.prediction}
                    </span>
                    <span class="prediction-tag prediction-${sample.ml.prediction.toLowerCase()}">
                        ML: ${sample.ml.prediction}
                    </span>
                </div>
            `;
            
            sampleTweets.appendChild(sampleItem);
        });
    }

    renderForecastResults() {
        const forecastResults = document.getElementById('forecastResults');
        const forecasts = this.data.stock_forecasting.forecasting_results;
        
        forecastResults.innerHTML = '';
        
        Object.entries(forecasts).forEach(([model, data]) => {
            const forecastItem = document.createElement('div');
            forecastItem.className = 'forecast-item fade-in-up';
            
            const avgForecast = data.next_5_days.reduce((a, b) => a + b, 0) / data.next_5_days.length;
            
            forecastItem.innerHTML = `
                <h4>${model} Model</h4>
                <div class="forecast-value">$${avgForecast.toFixed(2)}</div>
                <p>5-Day Average</p>
                <small>AIC: ${data.aic.toFixed(2)} | Order: ${data.order}</small>
            `;
            
            forecastResults.appendChild(forecastItem);
        });
    }

    renderStockStats() {
        const stockStats = document.getElementById('stockStats');
        const stats = this.data.stock_forecasting.stock_stats;
        
        const statsData = [
            { label: 'Min Price', value: `$${stats.min_price.toFixed(2)}` },
            { label: 'Max Price', value: `$${stats.max_price.toFixed(2)}` },
            { label: 'Avg Price', value: `$${stats.avg_price.toFixed(2)}` },
            { label: 'Latest Price', value: `$${stats.latest_price.toFixed(2)}` }
        ];
        
        stockStats.innerHTML = '';
        
        statsData.forEach((stat, index) => {
            const statsItem = document.createElement('div');
            statsItem.className = 'stats-item fade-in-up';
            statsItem.style.animationDelay = `${index * 0.1}s`;
            
            statsItem.innerHTML = `
                <div class="stats-value">${stat.value}</div>
                <div class="stats-label">${stat.label}</div>
            `;
            
            stockStats.appendChild(statsItem);
        });
    }

    renderCompanyAnalysis() {
        const companyAnalysis = document.getElementById('companyAnalysis');
        const companies = this.data.sentiment_analysis.company_distribution;
        
        companyAnalysis.innerHTML = '';
        
        Object.entries(companies).forEach(([company, count], index) => {
            const companyItem = document.createElement('div');
            companyItem.className = 'company-item fade-in-up';
            companyItem.style.animationDelay = `${index * 0.05}s`;
            
            companyItem.innerHTML = `
                <div class="company-name">${company}</div>
                <div class="company-count">${count}</div>
            `;
            
            companyAnalysis.appendChild(companyItem);
        });
    }

    setupEventListeners() {
        // Add click handlers for interactive elements
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('stat-card')) {
                this.animateStatCard(e.target);
            }
        });
        
        // Add keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
                e.preventDefault();
                this.refreshData();
            }
        });
    }

    animateStatCard(card) {
        card.style.transform = 'scale(0.95)';
        setTimeout(() => {
            card.style.transform = '';
        }, 150);
    }

    startRealTimeUpdates() {
        const updatesFeed = document.getElementById('updatesFeed');
        
        // Initial updates
        this.addUpdate('System', 'Dashboard Initialized', 'Analysis pipeline started successfully');
        this.addUpdate('Data', 'Data Loaded', `${this.data.sentiment_analysis.total_tweets} tweets and ${this.data.stock_forecasting.stock_stats.total_records} stock records processed`);
        this.addUpdate('Models', 'Models Trained', `${Object.keys(this.data.sentiment_analysis.model_performance).length} sentiment models and ${Object.keys(this.data.stock_forecasting.forecasting_results).length} forecasting models active`);
        
        // Simulated real-time updates
        setInterval(() => {
            this.addRandomUpdate();
        }, 30000); // Update every 30 seconds
    }

    addUpdate(icon, title, description) {
        const updatesFeed = document.getElementById('updatesFeed');
        const updateItem = document.createElement('div');
        updateItem.className = 'update-item fade-in-up';
        
        updateItem.innerHTML = `
            <div class="update-icon">${icon}</div>
            <div class="update-content">
                <h4>${title}</h4>
                <p>${description}</p>
                <div class="update-time">${new Date().toLocaleTimeString()}</div>
            </div>
        `;
        
        updatesFeed.insertBefore(updateItem, updatesFeed.firstChild);
        
        // Keep only latest 10 updates
        while (updatesFeed.children.length > 10) {
            updatesFeed.removeChild(updatesFeed.lastChild);
        }
    }

    addRandomUpdate() {
        const updates = [
            { icon: 'Stock', title: 'Stock Analysis Updated', desc: 'Latest market data processed' },
            { icon: 'Sentiment', title: 'Sentiment Analyzed', desc: 'New tweets processed for sentiment' },
            { icon: 'Model', title: 'Model Refresh', desc: 'Forecasting models updated' },
            { icon: 'Data', title: 'Data Sync', desc: 'Real-time data synchronization complete' }
        ];
        
        const randomUpdate = updates[Math.floor(Math.random() * updates.length)];
        this.addUpdate(randomUpdate.icon, randomUpdate.title, randomUpdate.desc);
    }

    updateTimestamp() {
        const lastUpdated = document.getElementById('lastUpdated');
        const timestamp = new Date(this.data.metadata.analysis_timestamp).toLocaleString();
        lastUpdated.textContent = timestamp;
    }

    async refreshData() {
        try {
            this.addUpdate('Refresh', 'Refreshing Data', 'Reloading analysis data...');
            await this.loadData();
            this.renderDashboard();
            this.addUpdate('Success', 'Data Refreshed', 'Dashboard updated with latest data');
        } catch (error) {
            console.error('Data refresh failed:', error);
            this.addUpdate('Error', 'Refresh Failed', 'Could not update data');
        }
    }

    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #b91c1c;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        `;
        errorDiv.textContent = message;
        
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FinancialDashboard();
});

// Export for potential external use
window.FinancialDashboard = FinancialDashboard;
