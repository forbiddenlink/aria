#!/bin/bash
# Deploy AI Artist Gallery to Vercel
# This script prepares and deploys the gallery-only mode to Vercel

set -e  # Exit on error

echo "🎨 AI Artist - Vercel Deployment Script"
echo "========================================"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found!"
    echo "Install it with: npm install -g vercel"
    exit 1
fi

echo "✅ Vercel CLI found: $(vercel --version)"
echo ""

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ vercel.json not found! Are you in the project root?"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo ""

# Option to optimize gallery
read -p "🔄 Do you want to optimize gallery images for web? (recommended) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing required packages..."
    pip install -q pillow tqdm

    echo "🔄 Optimizing gallery images..."
    python scripts/optimize_gallery_for_web.py

    if [ $? -eq 0 ]; then
        echo "✅ Gallery optimization complete!"
    else
        echo "⚠️  Gallery optimization failed, continuing with original images..."
    fi
    echo ""
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Creating one for Vercel..."
    cat > .env << EOF
GALLERY_ONLY_MODE=true
GALLERY_PATH=/tmp/gallery
DATABASE_URL=sqlite:////tmp/gallery.db
EOF
    echo "✅ Created .env file"
fi

echo "🔐 Vercel Login"
echo "If not logged in, you'll be prompted to authenticate..."
vercel whoami || vercel login

echo ""
echo "🔗 Linking project to Vercel..."
echo "If this is your first deployment, you'll be asked to configure the project."
echo ""

# Link or check if already linked
if [ ! -f ".vercel/project.json" ]; then
    echo "⚙️  Project not linked yet. Running vercel link..."
    vercel link
else
    echo "✅ Project already linked"
fi

echo ""
read -p "🚀 Ready to deploy to production? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "ℹ️  Deployment cancelled. You can deploy later with: vercel --prod"
    exit 0
fi

echo ""
echo "🚀 Deploying to Vercel..."
echo "This may take a few minutes..."
echo ""

# Deploy to production
vercel --prod

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Your gallery is now live!"
echo "🌐 Visit your deployment URL to see it in action"
echo ""
echo "💡 Next steps:"
echo "   - Share your gallery URL with others"
echo "   - Add more images by running: python -m ai_artist.main --mode manual"
echo "   - Set up automated generation: python -m ai_artist.main --mode auto"
echo "   - Customize your gallery in src/ai_artist/web/templates/"
echo ""
echo "🎨 Happy creating!"
