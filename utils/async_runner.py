import asyncio, threading, concurrent.futures, atexit
from typing import Awaitable, TypeVar, Optional

T = TypeVar("T")

class AsyncLoopThread:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Awaitable[T], timeout: Optional[float] = None) -> T:
        fut: concurrent.futures.Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def call_soon(self, fn, *args, **kwargs):
        self._loop.call_soon_threadsafe(fn, *args, **kwargs)

    def close(self):
        if getattr(self, "_loop", None) and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            try:
                self._loop.close()
            except Exception:
                pass

