from __future__ import annotations

import abc
import asyncio
import enum
import logging
import os
import subprocess
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, List, Optional, Union

import docker
import docker.errors
from docker.models.containers import Container
from docker.types import DeviceRequest

from .browser_runner import BrowserRunner
from .constants import (
    BASE_MESSAGING_URL,
    CAPTURE_FAILED_TIMEOUT_S,
    CAPTURE_SHELL_PATH,
    EXPERIENCE_DATA_PATH,
    PACKAGE_STATIC_PATH,
)
from .data.capture import CaptureApi, get_capture_api
from .data.video_devices import VideoDeviceManager, get_video_device_manager
from .util import mercilessly_kill_process

logger = logging.getLogger(__name__)

try:
    docker_client = docker.from_env()
except docker.errors.DockerException:
    docker_client = None
    logger.warning("Couldn't create Docker client, Docker experiences will be disabled")


class EnvironmentInitializationError(Exception):
    pass


class EnvironmentStateTransitionError(Exception):
    from_state: EnvironmentState
    to_state: EnvironmentState

    def __init__(self, from_state: EnvironmentState, to_state: EnvironmentState):
        self.from_state = from_state
        self.to_state = to_state
        super().__init__(
            f"Invalid environment state transition from '{from_state.name}' to '{to_state.name}'"
        )


class EnvironmentState(enum.Enum):
    """
    Valid state transitions:

    idle → starting → running → stopping → stopped
              ↓          ↓         ↓          ↓
              ╰─┬────────┴─────────┴──────────╯
                ↓
              failed

    """

    IDLE = enum.auto()
    STARTING = enum.auto()
    RUNNING = enum.auto()
    STOPPING = enum.auto()
    STOPPED = enum.auto()
    FAILED = enum.auto()


class BaseEnvironment(
    abc.ABC,
):
    _state: EnvironmentState

    def __init__(self):
        self._state = EnvironmentState.IDLE

    async def _attempt_state_transition(
        self,
        from_states: List[EnvironmentState],
        transition_state: EnvironmentState,
        settled_state: EnvironmentState,
        fn: Callable[[], Awaitable[None]],
    ):
        state = await self.state()
        if state not in from_states:
            raise EnvironmentStateTransitionError(state, transition_state)
        self._state = transition_state
        try:
            await fn()
        except Exception:
            self._state = EnvironmentState.FAILED
            raise
        self._state = settled_state

    async def start(self, last_environment: Optional[BaseEnvironment] = None, **kwargs):
        await self._attempt_state_transition(
            [
                EnvironmentState.IDLE,
                EnvironmentState.STOPPING,
                EnvironmentState.STOPPED,
                # TODO: This is in violation of the state diagram. We should think about
                #  this a little bit and then probably just allow transitioning to
                #  STARTING/STOPPING from FAILED.
                EnvironmentState.FAILED,
            ],
            EnvironmentState.STARTING,
            EnvironmentState.RUNNING,
            lambda: self._start(last_environment, **kwargs),
        )

    async def stop(self, next_environment: Optional[BaseEnvironment] = None, **kwargs):
        await self._attempt_state_transition(
            # Not sure if this is okay
            [
                EnvironmentState.RUNNING,
                EnvironmentState.STARTING,
                EnvironmentState.FAILED,
            ],
            EnvironmentState.STOPPING,
            EnvironmentState.STOPPED,
            lambda: self._stop(next_environment, **kwargs),
        )

    @abc.abstractmethod
    async def _start(
        self, last_environment: Optional[BaseEnvironment] = None, **kwargs
    ):
        ...

    @abc.abstractmethod
    async def _stop(self, next_environment: Optional[BaseEnvironment] = None, **kwargs):
        ...

    @abc.abstractmethod
    async def state(self) -> EnvironmentState:
        ...

    # TODO: Make this a regular function, not a property getter
    #  See https://python.org/dev/peps/pep-0008/#designing-for-inheritance:
    #  > Avoid using properties for computationally expensive operations; the attribute notation makes the caller
    #  believe that access is (relatively) cheap.
    @property
    def available(self) -> bool:
        return True


class _BaseWebEnvironment(BaseEnvironment):
    _runner: BrowserRunner
    _static_path: Path
    _available: bool
    _id: str

    def __init__(self, id, path, routes, url):
        super().__init__()
        self._available = True
        self._static_path = Path(path) if not isinstance(path, Path) else path
        self._check_static_path()
        self._runner = BrowserRunner(id, routes, url)

    async def _start(
        self, last_environment=None, map_localhost_ip: Optional[str] = None
    ):
        await self._runner.start(map_localhost_ip)

    async def _stop(self, next_environment=None, **_):
        await self._runner.stop()

    def _check_static_path(self):
        if self._static_path.exists():
            return
        self._available = False
        raise EnvironmentInitializationError(
            f"Couldn't load static path for experience {self._id}"
            f" at path {self._static_path.absolute()}"
        )

    async def state(self) -> EnvironmentState:
        if self._state != EnvironmentState.RUNNING:
            return self._state

        return (
            EnvironmentState.RUNNING
            if self._runner.check_running()
            else EnvironmentState.FAILED
        )

    @property
    def available(self):
        return self._available


class WebEnvironment(_BaseWebEnvironment):
    def __init__(
        self, id: str, path: Union[str, Path], url: Optional[str] = "/", **kwargs
    ):
        super().__init__(id, path, {"/": path}, url, **kwargs)


class VideoEnvironment(_BaseWebEnvironment):
    def __init__(self, id: str, path: Union[str, Path], video_filename: str, **kwargs):
        super().__init__(
            id,
            path,
            {"/video": path, "/": PACKAGE_STATIC_PATH / "video-player"},
            f"/?url=/video/{video_filename}&posterUrl=/video/poster.jpg&id={id}",
            **kwargs,
        )


class DockerEnvironment(BaseEnvironment):
    _id: str
    _container: Optional[Container]
    _ip: Optional[str]
    _video_devices: VideoDeviceManager
    _host_network: Optional[int]
    _image_exists: Optional[bool]
    _data_path: Optional[Path]

    def __init__(
        self,
        id: str,
        image_id: str,
        host_network: bool = False,
    ):
        super().__init__()
        self._id = id
        self._image_id = image_id
        self._container = None
        self._ip = None
        self._video_devices = get_video_device_manager()
        self._host_network = host_network
        self._image_exists = None
        self._data_path = EXPERIENCE_DATA_PATH / image_id.replace(":", "_").replace(
            "/", "_"
        )
        self._data_path.mkdir(parents=True, exist_ok=True)

    async def _start(self, last_environment=None, **_):
        # For now, we will expose only our center webcam as /dev/video0 within
        # containers
        video_devices = [
            f"{path}:/dev/video{name}:rw"
            for name, path in self._video_devices.devices.items()
        ]
        network_config = {"network_mode": "host"} if self._host_network else {}
        self._container = docker_client.containers.run(
            self._image_id,
            detach=True,
            volumes={
                "/tmp/.X11-unix": {"bind": "/tmp/.X11-unix", "mode": "rw"},
                str(self._data_path): {"bind": "/localdata", "mode": "rw"},
            },
            remove=True,
            stdout=False,
            environment=[
                f"DISPLAY={os.environ['DISPLAY']}",
                "NVIDIA_DRIVER_CAPABILITIES=all",
                f"FT_MSG_URL={urllib.parse.urljoin(BASE_MESSAGING_URL, self._id)}",
            ],
            device_requests=[
                DeviceRequest(driver="nvidia", count=-1, capabilities=[["gpu"]])
            ],
            devices=[*video_devices],
            # Chromium needs these to work, per @wingated
            cap_add=["SYS_ADMIN"],
            shm_size="1g",
            **network_config,
        )
        self._container.reload()
        self._ip = self._container.attrs["NetworkSettings"]["IPAddress"]

    def _kill_container_checked(self, container: Container):
        container.reload()
        if container.status != "running":
            return

        try:
            container.kill()
        except docker.errors.APIError as e:
            logger.exception(
                f"Docker errored while trying to kill container for app ID {self._id}:"
            )

    async def shutdown_by_tag(self):
        matching_containers = docker_client.containers.list(
            filters={"ancestor": self._image_id, "status": "running"}
        )
        if not matching_containers:
            return
        logger.warning(
            f"Found live containers with image ID {self._image_id}, attempting to kill in 1s"
        )
        await asyncio.sleep(1)
        list(map(self._kill_container_checked, matching_containers))

    async def _stop(self, next_environment=None, **_):
        self._kill_container_checked(self._container)
        await self.shutdown_by_tag()
        self._container = None

    async def state(self) -> EnvironmentState:
        if self._state != EnvironmentState.RUNNING:
            return self._state

        try:
            self._container.reload()
            container_status = self._container.status
        except docker.errors.NotFound as e:
            logger.exception(
                f"Docker errored while trying to get state of container for app ID {self._id}"
            )
            return EnvironmentState.FAILED

        if container_status in ["running", "created"]:
            return EnvironmentState.RUNNING

        return EnvironmentState.FAILED

    @property
    def available(self) -> bool:
        if not docker_client:
            return False
        if self._image_exists is not None:
            return self._image_exists

        try:
            docker_client.images.get(self._image_id)
            self._image_exists = True
            return True
        except docker.errors.ImageNotFound:
            logger.info(
                f"Couldn't find Docker image '{self._image_id}' locally, attempting to pull..."
            )
            try:
                docker_client.images.pull(self._image_id)
            except docker.errors.NotFound:
                logger.warning(
                    f"Couldn't find Docker image '{self._image_id}', experience will be disabled"
                )
                self._image_exists = False
                return False

            self._image_exists = True
            return True

    @property
    def ip(self) -> Optional[str]:
        return self._ip


class StackEnvironment(BaseEnvironment):
    _docker_environment: Optional[DockerEnvironment]
    _web_environment: Optional[WebEnvironment]

    def __init__(
        self,
        id: str,
        image_id: Optional[str],
        path: Optional[Union[str, Path]],
        url: Optional[str] = "/",
    ):
        if image_id is None and path is None:
            raise ValueError(
                "Must specify either 'image_id' or 'path' for stack experience"
            )

        super().__init__()
        self._docker_environment = DockerEnvironment(id, image_id) if image_id else None
        self._web_environment = WebEnvironment(id, path, url) if path else None

    async def _start(self, last_environment: Optional[BaseEnvironment] = None, **_):
        docker_ip = None
        if self._docker_environment:
            await self._docker_environment.start(last_environment)
            docker_ip = self._docker_environment.ip
            if docker_ip is None:
                raise ValueError("Couldn't read IP from Docker environment")
        if self._web_environment:
            await self._web_environment.start(
                last_environment, map_localhost_ip=docker_ip
            )

    async def _stop(self, next_environment: Optional[BaseEnvironment] = None, **_):
        if self._web_environment:
            await self._web_environment.stop(next_environment)
        if self._docker_environment:
            await self._docker_environment.stop(next_environment)

    async def state(self) -> EnvironmentState:
        docker_state = None
        if self._docker_environment:
            docker_state = await self._docker_environment.state()

        web_state = None
        if self._web_environment:
            web_state = await self._web_environment.state()

        if docker_state is None and web_state is None:
            # This shouldn't be possible
            return EnvironmentState.FAILED

        if docker_state and not web_state:
            return docker_state

        if web_state and not docker_state:
            return web_state

        if docker_state == web_state:
            return docker_state

        return EnvironmentState.FAILED


class CaptureEnvironment(BaseEnvironment):
    _id: str
    _path: str
    _load_time: Optional[int]

    _capture_process: Optional[subprocess.Popen]
    _api: CaptureApi
    _start_time: Optional[datetime]

    def __init__(self, id: str, path: str, load_time: Optional[int] = None):
        super().__init__()
        self._id = id
        self._path = path
        self._load_time = load_time

        self._api = get_capture_api()
        self._start_time = None

    async def _start_capture_api(self):
        await self._api.set_current_experience(self._id, self._path)
        self._start_time = datetime.now()

    async def _stop_capture_api(self):
        await self._api.set_current_experience(None)
        self._start_time = None

    async def _start_capture_process(self):
        self._capture_process = subprocess.Popen([CAPTURE_SHELL_PATH])

    async def _stop_capture_process(self):
        if not self._capture_process:
            return

        await mercilessly_kill_process(self._capture_process)

    async def _start(self, last_environment=None):
        await self._start_capture_api()
        await self._start_capture_process()

    async def _stop(self, next_environment=None):
        await self._stop_capture_process()
        if not next_environment or not isinstance(next_environment, CaptureEnvironment):
            await self._stop_capture_api()

    async def state(self) -> EnvironmentState:
        if self._state != EnvironmentState.RUNNING:
            return self._state

        capture_experience_response = await self._api.current_experience()
        capture_timeout = (
            max(self._load_time, CAPTURE_FAILED_TIMEOUT_S)
            if self._load_time
            else CAPTURE_FAILED_TIMEOUT_S
        )
        if not self._start_time or (
            (
                capture_experience_response.id is None
                or not capture_experience_response.processes
            )
            and (datetime.now() - self._start_time).seconds > capture_timeout
        ):
            return EnvironmentState.FAILED

        return EnvironmentState.RUNNING

    @property
    def available(self) -> bool:
        # TODO: Consider whether we need some availability signal for Windows apps
        return True
