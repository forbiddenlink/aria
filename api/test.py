"""Minimal test endpoint to debug Vercel deployment."""

def handler(event, context):
    """Simple handler that should work."""
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": '{"status": "working", "message": "Basic handler works!"}'
    }
