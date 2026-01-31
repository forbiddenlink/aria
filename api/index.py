"""Vercel serverless function - Simple HTTP handler."""

from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests."""
        path = self.path
        
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
            <h3>🖼️ Recent Artwork</h3>
            <div id="gallery" style="margin-top: 20px;">
                <p style="color: #999;">Loading artwork...</p>
            </div>
        </div>

        <div class="cta">
            <a href="https://github.com/forbiddenlink/aria" class="btn" target="_blank">View Full Collection on GitHub</a>
        </div>

        <footer>
            <p>Built with 💜 by Aria<br>
            <a href="/health">API Health Check</a></p>
        </footer>
    </div>

    <script>
        // Fetch gallery images from GitHub
        async function loadGallery() {
            const gallery = document.getElementById('gallery');
            try {
                // Fetch from GitHub API
                const response = await fetch('https://api.github.com/repos/forbiddenlink/aria/contents/gallery/2026/01/30/archive');
                const files = await response.json();
                
                // Filter for image files
                const images = files.filter(f => f.name.match(/\\.(jpg|jpeg|png|gif)$/i));
                
                if (images.length === 0) {
                    gallery.innerHTML = '<p style="color: #999;">No artwork available yet. Aria is preparing her first pieces!</p>';
                    return;
                }
                
                // Display images
                gallery.innerHTML = images.slice(0, 6).map(img => `
                    <div style="margin: 15px 0;">
                        <img src="${img.download_url}" 
                             alt="${img.name}" 
                             style="width: 100%; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
                             loading="lazy">
                        <p style="font-size: 0.85em; color: #999; margin-top: 8px;">${img.name.replace(/\\.[^/.]+$/, "").replace(/_/g, " ")}</p>
                    </div>
                `).join('');
                
            } catch (error) {
                console.error('Error loading gallery:', error);
                gallery.innerHTML = `
                    <p style="color: #999;">Gallery images available at:</p>
                    <a href="https://github.com/forbiddenlink/aria/tree/main/gallery" 
                       target="_blank" 
                       style="color: #667eea;">View on GitHub →</a>
                `;
            }
        }
        
        // Load gallery when page loads
        loadGallery();
    </script>
</body>
</html>"""
        self.wfile.write(html.encode())
