import logging
import subprocess
import urllib.parse
from pathlib import Path
from typing import Dict, Optional

from aiohttp import web
from aiohttp.web_log import AccessLogger
from aiohttp.web_runner import AppRunner, TCPSite

from .constants import BASE_BIN_PATH, BASE_MESSAGING_URL
from .util import find_free_port, mercilessly_kill_process

WEB_SHELL_PATH = BASE_BIN_PATH / "footron-web-shell"


class BrowserRunner:
    _id: str
    _url: str
    _port: int
    _app: web.Application
    _routes: Dict[str, str]
    _profile_path: Path
    _browser_process: Optional[subprocess.Popen]
    _runner = Optional[AppRunner]
    _site = Optional[TCPSite]

    def __init__(
        self,
        id: str,
        routes: Dict[str, str],
        url: str = "/",
    ):
        self._id = id
        self._app = web.Application(middlewares=[self.static_serve])
        self._routes = {route.rstrip("/"): path for route, path in routes.items()}
        self._url = url
        self._browser_process = None

    def _create_url(self):
        base_url = urllib.parse.urljoin(f"http://localhost", self._url)
        parsed_url = urllib.parse.urlsplit(base_url)
        query_params = urllib.parse.parse_qsl(parsed_url.query)
        query_params.append(
            ("ftMsgUrl", urllib.parse.urljoin(BASE_MESSAGING_URL, self._id))
        )
        query_str = urllib.parse.urlencode(query_params)
        return urllib.parse.urlunsplit(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                query_str,
                parsed_url.fragment,
            )
        )

    def _start_browser(self, map_localhost_ip: Optional[str] = None):
        host_rules = f"MAP localhost:80 localhost:{self._port}"
        if map_localhost_ip:
            host_rules += f", MAP localhost {map_localhost_ip}"
        host_rules_arg = f"--host-rules={host_rules}"
        self._browser_process = subprocess.Popen(
            [WEB_SHELL_PATH, self._create_url(), host_rules_arg]
        )

    async def _stop_browser(self):
        if not self._browser_process:
            return

        await mercilessly_kill_process(self._browser_process)

    def check_running(self):
        if self._browser_process.poll() is not None:
            return False
        # TODO: Figure out how to check whether our static server is running, though
        #  maybe we just handle exceptions?
        return True

    # Based on https://github.com/aio-libs/aiohttp/issues/1220#issuecomment-546572413
    @web.middleware
    async def static_serve(self, request, **kwargs):
        matching_route, root_path = next(
            (route, Path(path))
            for route, path in self._routes.items()
            if request.path.startswith(route)
        )
        relative_file_path = Path(
            request.path.replace(matching_route, "", 1)
        ).relative_to("/")
        file_path = root_path / relative_file_path
        if not file_path.exists():
            return web.HTTPNotFound()
        if file_path.is_dir():
            file_path /= "index.html"
            if not file_path.exists():
                return web.HTTPNotFound()
        return web.FileResponse(file_path)

    async def _start_static_server(self):
        self._port = find_free_port()
        self._runner = AppRunner(
            self._app,
            handle_signals=True,
            access_log_class=AccessLogger,
            access_log_format=AccessLogger.LOG_FORMAT,
            access_log=logging.getLogger(__name__),
        )

        await self._runner.setup()

        self._site = TCPSite(
            self._runner,
            port=self._port,
        )

        await self._site.start()

    async def _stop_static_server(self):
        if not self._runner or not self._site:
            return

        try:
            await self._runner.cleanup()
            logging.info("Cleaned up website")
        except RuntimeError as e:
            logging.error("Error while stopping static server:")
            logging.exception(e)

    async def start(self, map_localhost_ip: Optional[str] = None):
        await self._start_static_server()
        self._start_browser(map_localhost_ip)

    async def stop(self):
        await self._stop_browser()
        await self._stop_static_server()
