import asyncio
import datetime
import logging
import os
from typing import Dict, Optional

import aiohttp.client_exceptions
import footron_protocol as protocol
import rollbar

from .data.stability import StabilityManager
from .constants import EMPTY_EXPERIENCE_DATA
from .experiences import load_experiences_fs, BaseExperience
from .data.wm import WmApi
from .data.placard import PlacardApi, PlacardExperienceData
from .data.collection import load_collections_from_fs, Collection
from .data.tags import load_tags_from_fs, Tag

logger = logging.getLogger(__name__)


class Controller:
    experiences: Dict[str, BaseExperience] = {}
    collections: Dict[str, Collection] = {}
    tags: Dict[str, Tag] = {}
    collection_dictionary : Dict[str, str] = {}
    tag_dictionary : Dict[str, list] = {}
    current_experience: Optional[BaseExperience]
    current_experience_start: Optional[datetime.datetime]
    end_time: Optional[int]
    lock: protocol.Lock
    last_update: datetime.datetime
    placard: PlacardApi
    stability: StabilityManager
    _experience_modify_lock: asyncio.Lock

    def __init__(self):
        self.current_experience = None
        self.current_experience_start = None
        self.end_time = None
        self.lock = False
        self._experience_modify_lock = asyncio.Lock()

        self.placard = PlacardApi()
        self.wm = WmApi()
        self.stability = StabilityManager()

        self.load_from_fs()
        asyncio.get_event_loop().create_task(self.set_experience(None))

    def load_from_fs(self):
        self.load_experiences()
        self.load_collections()
        self.load_tags()
        self.fill_collection_dictionary()
        self.fill_tag_dictionary()
        self.last_update = datetime.datetime.now()

    def load_experiences(self):
        self.experiences = {
            experience.id: experience for experience in load_experiences_fs()
        }
        # for experience in self.experiences:
        #     for id, collection in self.collections.items():
        #         if experience in collection.experiences:
        #             experience.collection = collection.id
        #             break
            
        #     for tag in self.tags:
        #         if experience in tag.experiences:
        #             experience.tags.append(tag.id)


    def load_collections(self):
        self.collections = {
            collection.id: collection for collection in load_collections_from_fs()
        }
        print(self.collections)

    def load_tags(self):
        self.tags = {
            tag.id: tag for tag in load_tags_from_fs()
        }

    def fill_collection_dictionary(self):
        self.collection_dictionary = {}
        for collection in self.collections.values():
            for experience in collection.experiences:
                self.collection_dictionary[experience] = collection.id

    def fill_tag_dictionary(self):
        self.tag_dictionary = {}
        for experience in self.experiences:
            self.tag_dictionary[experience] = []
        for tag in self.tags.values():
            for experience in tag.experiences:
                if experience not in self.tag_dictionary:
                    self.tag_dictionary[experience] = [] 

                self.tag_dictionary[experience].append(tag.id)

    async def _update_experience_display(self, experience: Optional[BaseExperience]):
        # We don't actually want to wait for this to complete
        asyncio.get_event_loop().create_task(self._update_placard(experience))
        await self.wm.set_fullscreen(experience.fullscreen if experience else False)

    async def set_experience(self, id: Optional[str]):
        if self._experience_modify_lock.locked():
            return False

        await self._set_experience_impl(id)
        return True

    async def _set_experience_impl(self, id: Optional[str]):
        if self.current_experience and self.current_experience.id == id:
            return
        self.current_experience_start = datetime.datetime.now()

        # Unchecked exception, consumer's responsibility to know that experience with
        # ID exists
        experience = self.experiences[id] if id else None
        await self._update_experience_display(experience)

        try:
            await self.wm.clear_viewport()
            if self.current_experience:
                await self.current_experience.stop()
        finally:
            try:
                # Hang around to wait for windows to close without closing current
                # window
                await asyncio.sleep(1)
                if experience:
                    await experience.start()
            finally:
                # Environment start() and stop() methods should have their own error
                # handling, but if something is unhandled we need keep our state
                # maintained
                self.end_time = None
                self.lock = False
                self.current_experience = experience

    async def _update_placard(self, experience: BaseExperience):
        # TODO: Validate this worked somehow
        try:
            await self.placard.set_experience(
                PlacardExperienceData(
                    title=experience.title,
                    description=experience.description,
                    artist=experience.artist,
                )
                if experience
                else EMPTY_EXPERIENCE_DATA
            )
            await self.placard.set_visibility(
                not experience.fullscreen if experience else True
            )
        except aiohttp.client_exceptions.ClientError:
            logger.warning(
                "Updating placard failed with client exception, retrying in 1s"
            )
            # Wait for a second and try again
            await asyncio.sleep(1)
            await self._update_placard(experience)

    async def stability_loop(self):
        while True:
            logging.debug("Checking system stability...")
            try:
                if not self.stability.check_stable():
                    rollbar.report_message("System is unstable, rebooting")
                    logging.error("System is unstable, rebooting")
                    # Note that the current user has to have NOPASSWD set up in
                    # /etc/sudoers for /sbin/reboot on Ubuntu systems for this to
                    # work from Python
                    os.system("sudo reboot")
            except Exception as e:
                rollbar.report_exc_info(e)
                logger.exception(e)
            await asyncio.sleep(15)
