#!/bin/bash
set -e

echo "🚀 Setting up Quantonium OS..."
echo "================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version: $python_version"

if [[ $(echo "$python_version < 3.11" | bc -l) ]]; then
    echo "❌ Python 3.11+ required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Run basic validation
echo "🧪 Running validation tests..."
python tests/test_rft_roundtrip.py > /dev/null 2>&1 && echo "✅ RFT tests passed" || echo "⚠️  RFT tests had issues"
python tests/test_geowave_kat.py > /dev/null 2>&1 && echo "✅ Geometric tests passed" || echo "⚠️  Geometric tests had issues"
python test_quantonium_analysis.py > /dev/null 2>&1 && echo "✅ Analysis tests passed" || echo "⚠️  Analysis tests had issues"

# Check if main.py exists and can be imported
echo "🔍 Checking main application..."
python -c "import main; print('✅ Main application ready')" 2>/dev/null || echo "⚠️  Main application needs configuration"

echo ""
echo "🎉 Setup complete!"
echo "================================="
echo "To start the application:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo ""
echo "Or use the production server:"
echo "  gunicorn --bind 0.0.0.0:5000 main:app"
echo ""
echo "Access the web interface at: http://localhost:5000"
echo "View API docs at: http://localhost:5000/docs"
echo "================================="
