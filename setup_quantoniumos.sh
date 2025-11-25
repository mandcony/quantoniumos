#!/bin/bash
# Quick setup script for QuantoniumOS

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         QuantoniumOS Quick Setup                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python3 --version || { echo "❌ Python 3 not found!"; exit 1; }

# Install PyQt5
echo ""
echo "📦 Installing PyQt5 (this may take a moment)..."
pip3 install --user PyQt5 || { echo "⚠️  PyQt5 installation failed, but continuing..."; }

# Check if psutil is installed (for system monitor)
echo ""
echo "📦 Installing psutil for system monitor..."
pip3 install --user psutil || { echo "⚠️  psutil installation optional"; }

# Test imports
echo ""
echo "🧪 Testing core imports..."
python3 -c "from algorithms.rft.core.closed_form_rft import rft_forward" && echo "  ✓ RFT core" || echo "  ✗ RFT core failed"
python3 -c "from algorithms.rft.variants.registry import VARIANTS" && echo "  ✓ Transform variants" || echo "  ✗ Transform variants failed"
python3 -c "from algorithms.rft.crypto.enhanced_cipher import EnhancedRFTCryptoV2" && echo "  ✓ Crypto engine" || echo "  ✗ Crypto engine failed"
python3 -c "from algorithms.rft.compression.rft_vertex_codec import RFTVertexCodec" && echo "  ✓ Compression codec" || echo "  ✗ Compression codec failed"

# Test middleware
echo ""
echo "🔄 Testing middleware transform engine..."
python3 quantonium_os_src/engine/middleware_transform.py || { echo "⚠️  Middleware test had issues"; }

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo ""
echo "🚀 Launch QuantoniumOS with:"
echo "   python3 scripts/quantonium_boot.py"
echo ""
echo "Or test desktop directly:"
echo "   python3 quantonium_os_src/frontend/quantonium_desktop.py"
echo ""
echo "Or run validation tests:"
echo "   python3 scripts/test_desktop.py"
echo "═══════════════════════════════════════════════════════════"
