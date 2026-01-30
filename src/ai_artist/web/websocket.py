"""WebSocket manager for real-time updates."""

import asyncio
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts.

    Follows FastAPI WebSocket best practices for connection management.
    Thread-safe using asyncio.Lock for connection list modifications.
    """

    def __init__(self):
        # Use list instead of set for better iteration safety
        self.active_connections: list[WebSocket] = []
        self.generation_sessions: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, client_id: str = ""):
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(
            "websocket_connected",
            client_id=client_id,
            total_connections=len(self.active_connections),
        )

    async def disconnect(self, websocket: WebSocket, client_id: str = ""):
        """Remove a WebSocket connection."""
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(
            "websocket_disconnected",
            client_id=client_id,
            total_connections=len(self.active_connections),
        )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to a specific connection."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.warning("websocket_send_failed", error=str(e))
            # Remove dead connection
            await self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients.

        Handles disconnections gracefully and removes stale connections.
        Thread-safe: creates copy of connections list before iterating.
        """
        # Create copy to avoid modification during iteration
        async with self._lock:
            connections_copy = self.active_connections.copy()

        disconnected = []
        for connection in connections_copy:
            try:
                await connection.send_json(message)
            except (WebSocketDisconnect, RuntimeError) as e:
                # Connection closed or stale
                logger.debug("broadcast_connection_closed", error=str(e))
                disconnected.append(connection)
            except Exception as e:
                logger.warning("broadcast_failed", error=str(e))
                disconnected.append(connection)

        # Clean up disconnected clients
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)

    async def send_generation_progress(
        self, session_id: str, step: int, total_steps: int, message: str = ""
    ):
        """Send generation progress update."""
        progress = {
            "type": "generation_progress",
            "session_id": session_id,
            "step": step,
            "total_steps": total_steps,
            "progress_percent": int((step / total_steps) * 100),
            "message": message,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(progress)

    async def send_generation_complete(
        self, session_id: str, image_paths: list, metadata: dict
    ):
        """Send generation complete notification."""
        complete = {
            "type": "generation_complete",
            "session_id": session_id,
            "image_paths": image_paths,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(complete)

    async def send_generation_error(self, session_id: str, error: str):
        """Send generation error notification."""
        error_msg = {
            "type": "generation_error",
            "session_id": session_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(error_msg)

    async def send_curation_update(
        self, session_id: str, image_path: str, metrics: dict
    ):
        """Send curation metrics update."""
        update = {
            "type": "curation_update",
            "session_id": session_id,
            "image_path": image_path,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(update)

    async def send_gallery_update(self, action: str, image_data: dict):
        """Send gallery update notification."""
        update = {
            "type": "gallery_update",
            "action": action,  # "new_image", "deleted", "featured"
            "data": image_data,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(update)


# Global instance
manager = ConnectionManager()
