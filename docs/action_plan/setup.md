cd ~/ai-projects
mkdir stock-forecaster
cd stock-forecaster

# Create structure
mkdir -p data/raw data/processed src/data src/models models api results/plots docs

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install pandas numpy matplotlib seaborn yfinance scikit-learn tensorflow
