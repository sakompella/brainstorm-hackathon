import asyncio
import websockets

class FeatureServer:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.clients: set = set()

    async def register(self, ws) -> None:
        self.clients.add(ws)

    async def unregister(self, ws) -> None:
        self.clients.discard(ws)

    async def handler(self, ws) -> None:
        await self.register(ws)
        try:
            async for _ in ws:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(ws)

    async def broadcast(self, message: str) -> None:
        if not self.clients:
            return
        await asyncio.gather(
            *[c.send(message) for c in list(self.clients)],
            return_exceptions=True,
        )

    async def run(self) -> None:
        async with websockets.serve(self.handler, self.host, self.port):
            print(f"[feature server] listening on ws://{self.host}:{self.port}")
            await asyncio.Future()
