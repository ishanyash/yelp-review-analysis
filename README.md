# Yelp Review Analysis

![Yelp Review Analysis](./assets/yelp_banner.png)

## Project Overview

This repository contains a comprehensive analysis system for Yelp restaurant reviews that uses NLP and classification methods to understand customer behaviour and spending patterns.

### Performance Metrics
- **Accuracy**: 92%
- **Reviews Analysed**: 10K+

## Features

- **Text Mining**: Extracts meaningful insights from review text
- **Classification Models**: Categorises reviews based on sentiment and content
- **Customer Behaviour Analysis**: Identifies patterns in customer preferences and spending
- **Interactive Visualisations**: Provides clear visual representation of analysis results
- **Sentiment Trend Analysis**: Tracks changes in customer sentiment over time

## Technologies Used

- **Text Mining**: Advanced text processing techniques
- **Classification**: Machine learning algorithms for categorisation
- **Customer Behaviour Analysis**: Pattern recognition in review data
- **Data Visualisation**: Interactive charts and graphs

## Dataset

This project utilises a dataset consisting of Yelp restaurant reviews spanning over 15 years, including approximately 20,000 reviews across 45 restaurants. Each review includes:
- Restaurant URL (Yelp store)
- Rating
- Date
- Plain text review content

## Repository Structure

```
yelp_review_analysis/
├── data/               # Dataset files
├── notebooks/          # Jupyter notebooks for analysis and modelling
├── scripts/            # Python scripts for data processing and model training
├── models/             # Trained model files
├── docs/               # Documentation files
└── assets/             # Images and other assets
```

## Getting Started

### Prerequisites
- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yelp_review_analysis.git

# Navigate to the project directory
cd yelp_review_analysis

# Install dependencies
pip install -r requirements.txt
```

### Usage

1. Run the data preprocessing script:
   ```bash
   python scripts/preprocess_data.py
   ```

2. Execute the exploratory data analysis notebook:
   ```bash
   jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
   ```

3. Train the classification models:
   ```bash
   python scripts/train_models.py
   ```

4. Generate insights:
   ```bash
   python scripts/generate_insights.py
   ```

## Results

The analysis achieves a 92% accuracy in classifying customer behaviour patterns across more than 10,000 reviews, providing valuable insights for restaurant owners and marketers.

## Acknowledgements

- Yelp for the review data platform
- Restaurant owners and customers who contributed reviews

## License

This project is licensed under the MIT License - see the LICENSE file for details.
