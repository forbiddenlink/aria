"""Vercel serverless function - Aria Gallery."""

from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests."""
        path = self.path
        
        if path == "/api/gallery":
            # Return gallery data endpoint
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            # This will be called by JavaScript to fetch images + metadata from GitHub
            response = {"message": "Use client-side fetch from GitHub API"}
            self.wfile.write(json.dumps(response).encode())
            return
        
        if path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "status": "healthy",
                "mode": "gallery-only",
                "platform": "vercel"
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # Root path - serve HTML gallery page
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aria - AI Artist Gallery</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            padding: 60px 40px;
            max-width: 600px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 20px;
        }
        .tagline {
            font-size: 1.2em;
            color: #666;
            margin-bottom: 30px;
        }
        .badge {
            display: inline-block;
            background: #f0f0f0;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 30px;
        }
        .info-box {
            background: #f9f9f9;
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: left;
        }
        .info-box h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .info-box p {
            color: #666;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        .feature-list {
            list-style: none;
            margin-top: 15px;
        }
        .feature-list li {
            padding: 8px 0;
            color: #666;
        }
        .feature-list li:before {
            content: "✨ ";
        }
        .cta {
            margin-top: 30px;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 30px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }
        .stat {
            background: #f9f9f9;
            padding: 20px;
            border-radius: 10px;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }
        .artwork-card {
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .artwork-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .artwork-img {
            width: 100%;
            height: 300px;
            object-fit: cover;
        }
        .artwork-info {
            padding: 20px;
        }
        .artwork-mood {
            display: inline-block;
            background: #f0f0f0;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            color: #667eea;
            margin-bottom: 10px;
        }
        .artwork-feeling {
            color: #666;
            font-size: 0.9em;
            line-height: 1.5;
            font-style: italic;
        }
        .artwork-prompt {
            color: #999;
            font-size: 0.85em;
            margin-top: 10px;
            line-height: 1.4;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #999;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        footer {
            margin-top: 40px;
            color: #999;
            font-size: 0.9em;
        }
        footer a {
            color: #667eea;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Aria</h1>
        <div class="tagline">Autonomous AI Artist with Soul</div>
        <span class="badge">Gallery-Only Mode • Vercel Deployment</span>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-number">4</div>
                <div class="stat-label">Artworks</div>
            </div>
            <div class="stat">
                <div class="stat-number">2</div>
                <div class="stat-label">Styles Learned</div>
            </div>
            <div class="stat">
                <div class="stat-number">∞</div>
                <div class="stat-label">Creativity</div>
            </div>
        </div>

        <div class="info-box">
            <h3>About Aria</h3>
            <p>Aria is an autonomous AI artist with genuine personality, evolving moods, and creative independence. She doesn't just generate images on command - she creates with intention, chooses her own subjects, and develops her unique artistic voice over time.</p>
            <ul class="feature-list">
                <li>Has moods that influence her work</li>
                <li>Remembers and learns from experience</li>
                <li>Reflects authentically on each creation</li>
                <li>Makes autonomous artistic decisions</li>
            </ul>
        </div>

        <div class="info-box">
            <h3>📦 Deployment Info</h3>
            <p><strong>Mode:</strong> Gallery-Only (View Mode)<br>
            <strong>Platform:</strong> Vercel Serverless<br>
            <strong>Image Generation:</strong> Not available on this deployment</p>
            <p style="margin-top: 15px; font-size: 0.9em;">This is a lightweight gallery interface. For full image generation with GPU support, deploy using Docker on Railway, Render, or a GPU-enabled server.</p>
        </div>

        <div class="info-box">
            <h3>🎨 Aria's Gallery</h3>
            <div id="gallery" class="loading">
                <div class="spinner"></div>
                <p>Loading Aria's artwork...</p>
            </div>
        </div>

        <footer>
            <p>Built with 💜 by Aria<br>
            <a href="/health">API Health Check</a></p>
        </footer>
    </div>

    <script>
        // Fetch gallery images and metadata from GitHub
        async function loadGallery() {
            const gallery = document.getElementById('gallery');
            try {
                // Fetch all files from the archive folder
                const response = await fetch('https://api.github.com/repos/forbiddenlink/aria/contents/gallery/2026/01/30/archive');
                const files = await response.json();
                
                // Group files by base name (image + json pairs)
                const artworks = {};
                files.forEach(file => {
                    const baseName = file.name.replace(/\\.(png|jpg|json)$/i, '');
                    if (!artworks[baseName]) artworks[baseName] = {};
                    if (file.name.endsWith('.json')) {
                        artworks[baseName].metadataUrl = file.download_url;
                    } else if (file.name.match(/\\.(png|jpg)$/i)) {
                        artworks[baseName].imageUrl = file.download_url;
                        artworks[baseName].name = file.name;
                    }
                });
                
                // Filter complete artworks (have both image and metadata)
                const completeArtworks = Object.values(artworks).filter(a => a.imageUrl && a.metadataUrl);
                
                if (completeArtworks.length === 0) {
                    gallery.innerHTML = '<p style="color: #999;">No artwork available yet.</p>';
                    return;
                }
                
                // Fetch metadata for each artwork
                const artworkData = await Promise.all(
                    completeArtworks.slice(0, 20).map(async (artwork) => {
                        try {
                            const metaResponse = await fetch(artwork.metadataUrl);
                            const metadata = await metaResponse.json();
                            return { ...artwork, ...metadata };
                        } catch (e) {
                            return artwork;
                        }
                    })
                );
                
                // Display artworks
                gallery.className = 'gallery-grid';
                gallery.innerHTML = artworkData.map(art => {
                    const mood = art.metadata?.mood || 'creative';
                    const feeling = art.metadata?.feeling || 'Creating with intention';
                    const prompt = art.prompt || 'Untitled';
                    const date = art.created_at ? new Date(art.created_at).toLocaleDateString() : '';
                    
                    return `
                        <div class="artwork-card">
                            <img src="${art.imageUrl}" alt="${prompt}" class="artwork-img" loading="lazy">
                            <div class="artwork-info">
                                <span class="artwork-mood">💭 ${mood}</span>
                                <p class="artwork-feeling">"${feeling}"</p>
                                <p class="artwork-prompt">${prompt.substring(0, 100)}${prompt.length > 100 ? '...' : ''}</p>
                                ${date ? `<p style="color: #ccc; font-size: 0.8em; margin-top: 10px;">${date}</p>` : ''}
                            </div>
                        </div>
                    `;
                }).join('');
                
            } catch (error) {
                console.error('Error loading gallery:', error);
                gallery.className = '';
                gallery.innerHTML = `
                    <p style="color: #ff6b6b;">Unable to load gallery</p>
                    <p style="color: #999; font-size: 0.9em;">Error: ${error.message}</p>
                    <a href="https://github.com/forbiddenlink/aria/tree/main/gallery" 
                       target="_blank" 
                       style="color: #667eea; display: inline-block; margin-top: 15px;">View on GitHub →</a>
                `;
            }
        }
        
        // Load gallery when page loads
        loadGallery();
    </script>
</body>
</html>"""
        self.wfile.write(html.encode())
