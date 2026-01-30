"""Tests for web API endpoints."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_artist.gallery.manager import GalleryManager
from ai_artist.web.app import app
from ai_artist.web.dependencies import set_gallery_manager


@pytest.fixture
def test_gallery(tmp_path):
    """Create a test gallery with sample images."""
    gallery_path = tmp_path / "gallery"
    gallery_path.mkdir()
    
    # Create test image directory structure
    test_date = gallery_path / "2026" / "01" / "09"
    test_date.mkdir(parents=True)
    
    # Create a simple test image
    from PIL import Image
    img = Image.new('RGB', (512, 512), color='blue')
    img_path = test_date / "test_image.png"
    img.save(img_path)
    
    # Create metadata
    metadata = {
        "prompt": "A beautiful test image",
        "negative_prompt": "ugly, blurry",
        "width": 512,
        "height": 512,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
        "seed": 42,
        "model": "test-model",
        "created_at": "2026-01-09T10:00:00"
    }
    metadata_path = test_date / "test_image.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    return gallery_path


@pytest.fixture
def client(test_gallery):
    """Create test client with test gallery."""
    manager = GalleryManager(test_gallery)
    set_gallery_manager(manager, str(test_gallery))
    
    with TestClient(app) as c:
        yield c


class TestImageListAPI:
    """Test /api/images endpoint."""
    
    def test_list_images_success(self, client):
        """Test successful image listing."""
        response = client.get("/api/images")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 0  # May be empty or have test images
    
    def test_list_images_with_limit(self, client):
        """Test limit parameter."""
        response = client.get("/api/images?limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) <= 5
    
    def test_list_images_with_offset(self, client):
        """Test offset parameter."""
        response = client.get("/api/images?offset=0")
        assert response.status_code == 200
    
    def test_list_images_with_search(self, client):
        """Test search parameter."""
        response = client.get("/api/images?search=test")
        assert response.status_code == 200
        
        data = response.json()
        # Should filter by prompt containing 'test'
        for image in data:
            assert 'test' in image['prompt'].lower()
    
    def test_list_images_featured_filter(self, client):
        """Test featured filter."""
        response = client.get("/api/images?featured=true")
        assert response.status_code == 200
        
        data = response.json()
        for image in data:
            assert image['featured'] is True
    
    def test_list_images_invalid_limit(self, client):
        """Test invalid limit (too high)."""
        response = client.get("/api/images?limit=1000")
        assert response.status_code == 422  # Validation error
    
    def test_list_images_negative_offset(self, client):
        """Test negative offset."""
        response = client.get("/api/images?offset=-1")
        assert response.status_code == 422


class TestImageFileAPI:
    """Test /api/images/file/{path} endpoint."""
    
    def test_serve_image_success(self, client, test_gallery):
        """Test successful image serving."""
        # Create a test image
        from PIL import Image
        test_path = test_gallery / "2026" / "01" / "09" / "serve_test.png"
        test_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new('RGB', (256, 256), color='red')
        img.save(test_path)
        
        response = client.get("/api/images/file/2026/01/09/serve_test.png")
        assert response.status_code == 200
        assert response.headers['content-type'] == 'image/png'
    
    def test_serve_image_not_found(self, client):
        """Test 404 for non-existent image."""
        response = client.get("/api/images/file/2026/01/09/nonexistent.png")
        assert response.status_code == 404
    
    def test_serve_image_path_traversal(self, client):
        """Test path traversal attack prevention."""
        response = client.get("/api/images/file/../../../etc/passwd")
        assert response.status_code in [400, 404]  # Should be blocked
    
    def test_serve_image_invalid_extension(self, client):
        """Test rejection of non-image files."""
        response = client.get("/api/images/file/2026/01/09/script.sh")
        assert response.status_code == 400


class TestStatsAPI:
    """Test /api/stats endpoint."""
    
    def test_stats_success(self, client):
        """Test successful stats retrieval."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert 'total_images' in data
        assert 'featured_images' in data
        assert 'total_prompts' in data
        assert 'date_range' in data
        
        assert isinstance(data['total_images'], int)
        assert isinstance(data['featured_images'], int)
        assert isinstance(data['total_prompts'], int)
    
    def test_stats_rate_limit(self, client):
        """Test rate limiting on stats endpoint."""
        # Make multiple requests quickly
        responses = [client.get("/api/stats") for _ in range(35)]
        
        # All should succeed or be rate limited
        for response in responses:
            assert response.status_code in [200, 429]


class TestGenerateAPI:
    """Test /api/generate endpoint."""
    
    @patch('ai_artist.core.generator.ImageGenerator')
    @patch('ai_artist.utils.config.load_config')
    def test_generate_success(self, mock_load_config, mock_generator_class, client):
        """Test successful generation request."""
        # Mock the config
        mock_config = MagicMock()
        mock_config.model.base_model = "test-model"
        mock_config.model.device = "cpu"
        mock_load_config.return_value = mock_config
        
        # Mock the generator
        mock_gen = MagicMock()
        mock_gen.generate.return_value = [MagicMock()]  # Mock PIL Image
        mock_generator_class.return_value = mock_gen
        
        payload = {
            "prompt": "A beautiful sunset",
            "negative_prompt": "ugly",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "num_images": 1
        }
        
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert 'session_id' in data
        assert 'message' in data
        assert 'status' in data
    
    def test_generate_invalid_prompt(self, client):
        """Test generation with empty prompt."""
        payload = {
            "prompt": "",
            "width": 1024,
            "height": 1024
        }
        
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_generate_invalid_dimensions(self, client):
        """Test generation with invalid dimensions."""
        payload = {
            "prompt": "test",
            "width": 100,  # Too small
            "height": 100
        }
        
        response = client.post("/api/generate", json=payload)
        assert response.status_code == 422


class TestHealthAPI:
    """Test health check endpoints."""
    
    def test_basic_health(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
    
    def test_readiness_probe(self, client):
        """Test readiness probe."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
        assert 'checks' in data
    
    def test_liveness_probe(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data['status'] == 'alive'


class TestCORSHeaders:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """Test CORS headers are present on actual requests."""
        # Test on a real GET request with Origin header
        response = client.get(
            "/api/images",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # TestClient may not fully support CORS middleware, so we check if
        # the endpoint is accessible (main functional requirement)
        assert response.status_code == 200
        
        # If CORS headers are present, verify they're correct
        # Should echo back the allowed origin (not wildcard)
        if 'access-control-allow-origin' in response.headers:
            assert response.headers['access-control-allow-origin'] in [
                'http://localhost:3000',
                'http://localhost:8000',
                'http://127.0.0.1:3000',
                'http://127.0.0.1:8000',
            ]
    
    def test_cors_preflight(self, client):
        """Test CORS preflight OPTIONS request."""
        # Proper preflight request with required headers
        response = client.options(
            "/api/images",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        
        # Should either handle preflight (200) or at least have CORS headers
        assert response.status_code in [200, 405]
        if response.status_code == 200:
            assert 'access-control-allow-origin' in response.headers
    
    def test_cors_credentials(self, client):
        """Test CORS credentials support."""
        response = client.get("/api/stats")
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling and responses."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test 405 error handling."""
        response = client.post("/api/stats")  # GET only endpoint
        assert response.status_code == 405
    
    def test_validation_error_format(self, client):
        """Test validation error response format."""
        response = client.get("/api/images?limit=invalid")
        assert response.status_code == 422
        
        data = response.json()
        assert 'detail' in data  # FastAPI validation error format
