# Biznet Twitter Sentiment Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://biznet-sentiment-analysis.streamlit.app)

Try the live application: [Biznet Sentiment Analysis App](https://biznet-sentiment-analysis.streamlit.app)

![App Screenshot](data/figures/sentiment_distribution.png)

## Overview
This project analyzes sentiment from tweets related to Biznet (Indonesian Internet Service Provider) using Natural Language Processing techniques. The analysis covers tweets collected from June 19, 2025, to October 31, 2025.

## Features
- Sentiment analysis visualization with real-time updates
- Interactive dashboard with customizable filters
- Advanced keyword extraction and trend analysis
- Comprehensive data preprocessing pipeline
- Multi-dimensional visualization including word clouds and trend charts

## Technology Stack
- **Core**: Python, Streamlit
- **ML/NLP**: HuggingFace Transformers (IndoRoBERTa)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly, Matplotlib

## Installation

1. Clone the repository
```bash
git clone https://github.com/Fikri645/biznet-sentiment-analysis.git
cd biznet-sentiment-analysis
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run app/streamlit_app.py
```

## Project Structure
```
biznet-sentiment-analysis/
├── app/
│   └── streamlit_app.py      # Main Streamlit application
├── data/
│   ├── figures/              # Generated visualizations
│   ├── slang.csv            # Indonesian slang dictionary
│   └── *_public.csv         # Public dataset samples
├── src/
│   ├── analysis.py          # Analysis utilities
│   ├── preprocess.py        # Data preprocessing
│   ├── sentiment.py         # Sentiment analysis
│   └── utils.py             # Helper functions
└── requirements.txt         # Project dependencies
```

![Trend Analysis](data/figures/sentiment_trend.png)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
[Fikri Wahidin](https://github.com/Fikri645)

## Technical Details

### Data Collection
- **Source**: Twitter API via [TwitterAPI.io](https://twitterapi.io/)
- **Period**: June 19, 2025 - October 31, 2025
- **Query Parameters**:
  ```
  "Biznet" OR "#Biznet" lang:in -filter:links since:2025-06-19_00:00:00_UTC until:2025-10-31_23:59:59_UTC
  ```

### Data Privacy
- Anonymized dataset with sequential identifiers
- Preserved data structure while protecting user privacy
- Compliant with Twitter's Terms of Service

### Preprocessing Pipeline
1. Text cleaning and normalization
2. Slang word translation using comprehensive dictionary
3. Sentiment analysis using IndoRoBERTa
4. Visualization and trend analysis

### Dashboard Features
- **Main Dashboard**
  - Sentiment distribution
  - Temporal trend analysis
  - Key performance metrics
  - Interactive visualizations

- **Analysis Tools**
  - Keyword extraction and frequency analysis
  - TF-IDF term analysis
  - Bigram analysis with PMI scoring
  - Customizable filters and controls

- **Reporting**
  - Automated summary generation
  - Distribution analysis
  - Theme identification
  - Downloadable reports

### Development Setup

1. Environment preparation
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows
```

2. Installation
```bash
pip install -r requirements.txt
```

3. Data preparation
- Configure data files in `data/` directory
- Update slang dictionary if needed
- Run preprocessing pipeline:
  ```bash
  python src/preprocess.py
  python src/sentiment.py
  ```

### Requirements
- Python 3.8+
- Key packages: streamlit, pandas, scikit-learn, Sastrawi, plotly
- See `requirements.txt` for complete dependency list

## Contributing
Feel free to open issues or submit pull requests for improvements.
