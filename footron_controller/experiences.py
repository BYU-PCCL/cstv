from __future__ import annotations

import abc
import asyncio
import json
import operator
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Generic, List, Optional, Type, TypeVar

import footron_protocol as protocol
import tomli
from pydantic import BaseModel, PrivateAttr, root_validator, validator

from .constants import EXPERIENCES_PATH, VIDEO_ACTION_HINTS, JsonDict
from .data.wm import DisplayLayout
from .environments import (
    BaseEnvironment,
    CaptureEnvironment,
    DockerEnvironment,
    VideoEnvironment,
    WebEnvironment,
    StackEnvironment,
)

_DEFAULT_LIFETIME = 60
_FIELD_TYPE = "type"
_CONFIG_FILENAME = "config"


class ExperienceType(str, Enum):
    Stack = "stack"
    Docker = "docker"
    Web = "web"
    Video = "video"
    Capture = "capture"


EnvironmentType = TypeVar("EnvironmentType", bound=BaseEnvironment)


class BaseExperience(BaseModel, abc.ABC, Generic[EnvironmentType]):
    type: ExperienceType
    id: str
    title: str
    description: Optional[str]
    long_description: Optional[str]
    artist: Optional[str]
    lifetime: int = _DEFAULT_LIFETIME
    layout: DisplayLayout = DisplayLayout.Wide
    unlisted: bool = False
    queueable: bool = True
    load_time: Optional[int] = None
    action_hints: List[str] = []
    experience_path: Path
    _environment: EnvironmentType = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._environment = self._create_environment()

    @property
    def available(self) -> bool:
        return self._environment.available

    @property
    def environment(self) -> BaseEnvironment:
        return self._environment

    @validator("long_description")
    def long_description_requires_description(cls, value, values):
        if "description" not in values or values["description"] is None:
            raise ValueError(
                "'description' must be set for 'long_description' to be set"
            )
        return value

    async def start(self, last_experience: Optional[BaseExperience] = None):
        last_environment = last_experience._environment if last_experience else None
        await self._environment.start(last_environment)

    async def stop(
        self,
        next_experience: Optional[BaseExperience] = None,
        after: Optional[int] = None,
    ):
        if not self._environment:
            raise RuntimeError("Can't stop uninitialized environment")
        next_environment = next_experience._environment if next_experience else None
        if after:
            await asyncio.sleep(after)
        await self._environment.stop(next_environment)

    @abc.abstractmethod
    def _create_environment(self) -> BaseEnvironment:
        ...


class StackExperience(BaseExperience[StackEnvironment]):
    """Combines web and docker experiences"""

    type = ExperienceType.Stack
    image_id: Optional[str]
    url: Optional[str]
    layout = DisplayLayout.Wide

    @root_validator(pre=True)
    def validate_has_environment(cls, values):
        if (
            "docker_image" not in values
            and not (values["experience_path"] / "static").exists()
        ):
            raise ValueError(
                "'docker_image' parameter or 'static' directory required for stack"
            )
        return values

    def _create_environment(self) -> BaseEnvironment:
        static_path = self.experience_path / "static"
        return StackEnvironment(
            self.id,
            self.image_id,
            static_path if static_path.exists() else None,
            self.url,
        )


class DockerExperience(BaseExperience[DockerEnvironment]):
    type = ExperienceType.Docker
    image_id: str
    host_network: bool = False
    layout = DisplayLayout.Wide

    def _create_environment(self) -> DockerEnvironment:
        return DockerEnvironment(self.id, self.image_id, self.host_network)

    async def attempt_cleanup(self):
        await self._environment.shutdown_by_tag()


class WebExperience(BaseExperience[WebEnvironment]):
    type = ExperienceType.Web
    url: Optional[str]
    layout = DisplayLayout.Wide

    def _create_environment(self) -> WebEnvironment:
        return WebEnvironment(self.id, self.experience_path / "static", self.url)


class VideoExperience(BaseExperience[VideoEnvironment]):
    type = ExperienceType.Video
    layout = DisplayLayout.Hd
    filename: str
    scrubbing: bool = False

    def _create_environment(self) -> VideoEnvironment:
        return VideoEnvironment(self.id, self.experience_path, self.filename)

    @root_validator(pre=True)
    def validate_video_action_hints(cls, values):
        if "action_hints" in values:
            raise ValueError("Can't set 'action_hints' for video experiences")
        if "scrubbing" in values and values["scrubbing"]:
            values["action_hints"] = VIDEO_ACTION_HINTS
        return values


class CaptureExperience(BaseExperience[CaptureEnvironment]):
    type = ExperienceType.Capture
    layout = DisplayLayout.Full
    path: str

    def _create_environment(self) -> CaptureEnvironment:
        return CaptureEnvironment(self.id, self.path, self.load_time)


class Lock:
    status: protocol.Lock
    last_update: Optional[datetime]

    def __init__(self, status: protocol.Lock, last_update: Optional[datetime]):
        self.status = status
        self.last_update = last_update


class CurrentExperience:
    _experience: BaseExperience
    _start_time: datetime
    _lock: Lock
    end_time: Optional[datetime]
    last_interaction: Optional[datetime]

    def __init__(self, experience: BaseExperience, start_time: datetime):
        self._experience = experience
        self._start_time = start_time
        self._lock = Lock(False, None)
        self.end_time = None
        self.last_interaction = None

    @property
    def id(self):
        return self._experience.id

    @property
    def experience(self):
        return self._experience

    @property
    def environment(self):
        return self._experience.environment if self._experience else None

    @property
    def start_time(self):
        return self._start_time

    @property
    def lock(self):
        return self._lock

    @lock.setter
    def lock(self, value: protocol.Lock):
        if type(value) != int and type(value) != bool:
            raise ValueError("lock value must be of type int or bool")

        # Setting the value to what it was before doesn't represent
        # an "update," and we don't want to store a new last update
        # time. This is especially true because we use a non-null value
        # in that field as "breaking the seal" on a lock—any time we
        # set and unset a lock, we want to immediately cycle to the next
        # experience.
        if self._lock.status == value:
            return

        self._lock.status = value
        self._lock.last_update = datetime.now()

    async def start(self):
        return await self._experience.start()

    async def stop(self, after: Optional[int] = None):
        return await self._experience.stop(after)

    async def stop_(self):
        return await self._experience.stop()


experience_type_map: Dict[ExperienceType, Type[BaseExperience]] = {
    ExperienceType.Stack: StackExperience,
    ExperienceType.Docker: DockerExperience,
    ExperienceType.Web: WebExperience,
    ExperienceType.Video: VideoExperience,
    ExperienceType.Capture: CaptureExperience,
}


def _deserialize_experience(data: JsonDict, path: Path) -> BaseExperience:
    if _FIELD_TYPE not in data:
        raise TypeError(f"Experience doesn't contain required field '{_FIELD_TYPE}'")

    if not isinstance(data[_FIELD_TYPE], ExperienceType):
        data[_FIELD_TYPE] = ExperienceType(data[_FIELD_TYPE])

    msg_type: ExperienceType = data[_FIELD_TYPE]

    return experience_type_map[msg_type](**data, experience_path=path)


def _load_config_at_path(path: Path):
    json_path = path / f"{_CONFIG_FILENAME}.json"
    toml_path = path / f"{_CONFIG_FILENAME}.toml"

    json_exists = json_path.exists()
    toml_exists = toml_path.exists()

    if not json_exists and not toml_exists:
        # TODO: Log here instead of silently ignoring experiences without configs
        return

    config_path = json_path if json_exists else toml_path

    try:
        with open(config_path, "rb") as config_file:
            config = (json if json_exists else tomli).load(config_file)
        return config
    except (ValueError, tomli.TOMLDecodeError):
        print(
            f"Failed to parse config at path '{config_path}'",
            file=sys.stderr,
        )
        return


def _load_experience_at_path(path: Path) -> Optional[BaseExperience]:
    if not path.is_dir():
        return

    return _deserialize_experience(_load_config_at_path(path), path)


def load_experiences_fs(path=EXPERIENCES_PATH):
    return list(
        filter(
            operator.attrgetter("available"),
            map(_load_experience_at_path, path.iterdir()),
        )
    )
