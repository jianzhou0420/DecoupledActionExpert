"""WebSocket policy server for TaskFusion policies."""

import asyncio
import http
import logging
import time
import traceback

import msgpack
import numpy as np
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def pack_array(obj):
    """Serialize numpy arrays for msgpack."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    return obj


def unpack_array(obj):
    """Deserialize numpy arrays from msgpack."""
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    return obj


class WebsocketPolicyServer:
    """Serves a TaskFusion policy via WebSocket.

    This server accepts observations from clients, runs policy inference,
    and returns predicted actions.

    Example:
        wrapper = TaskFusionPolicyWrapper(policy, device="cuda:0")
        server = WebsocketPolicyServer(wrapper, host="0.0.0.0", port=8000)
        server.serve_forever()
    """

    def __init__(self, policy, host: str = "0.0.0.0", port: int = 8000, metadata: dict = None):
        """Initialize the server.

        Args:
            policy: Policy object with infer(obs) -> action method
            host: Host address to bind to
            port: Port number to listen on
            metadata: Optional metadata to send to clients on connect
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self):
        """Start the server and block forever."""
        asyncio.run(self.run())

    async def run(self):
        """Async server main loop."""
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"Server listening on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket):
        """Handle a single client connection."""
        logger.info(f"Connection from {websocket.remote_address}")
        packer = msgpack.Packer(default=pack_array)

        # Send metadata on connect
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None

        while True:
            try:
                start_time = time.monotonic()

                # Receive observation
                raw = await websocket.recv()
                obs = msgpack.unpackb(raw, object_hook=unpack_array)

                # Run inference (batch or single)
                # Note: msgpack unpacks keys as bytes by default
                infer_start = time.monotonic()
                if isinstance(obs, dict) and (obs.get("__batch__") or obs.get(b"__batch__")):
                    # Batch inference
                    observations = obs.get("observations") or obs.get(b"observations")
                    results = self._policy.infer_batch(observations)
                    action = {
                        "__batch__": True,
                        "results": results,
                    }
                else:
                    # Single-sample inference (backward compatible)
                    action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_start

                # Add timing info
                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                # Send action
                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection closed: {websocket.remote_address}")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal error",
                )
                raise

    @staticmethod
    def _health_check(conn, req):
        """Handle health check requests."""
        if req.path == "/healthz":
            return conn.respond(http.HTTPStatus.OK, "OK\n")
        return None
