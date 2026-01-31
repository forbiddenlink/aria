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
        
        # Root path
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        response = {
            "name": "Aria - AI Artist Gallery",
            "mode": "gallery-only",
            "message": "Gallery-only deployment. For full generation, use Docker."
        }
        self.wfile.write(json.dumps(response).encode())
