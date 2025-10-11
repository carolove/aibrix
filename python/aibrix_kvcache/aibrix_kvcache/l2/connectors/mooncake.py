# Copyright 2024 The Aibrix Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# 	http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import Executor
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple
import threading

import torch

from ... import envs
from ...common import AsyncBase
from ...common.absl_logging import getLogger
from ...memory import MemoryRegion
from ...status import Status, StatusCodes
from . import Connector, ConnectorFeature

logger = getLogger(__name__)


@dataclass
class MooncakeConfig:
    local_hostname: str
    metadata_server: str
    global_segment_size: int
    local_buffer_size: int
    protocol: str
    rdma_devices: str
    master_server_addr: str


class MooncakeConnector(Connector[bytes, torch.Tensor], AsyncBase):
    """Mooncake connector."""

    def __init__(
        self,
        config: MooncakeConfig,
        key_suffix: str,
        executor: Executor,
    ):
        super().__init__(executor)
        self.config = config
        self.key_suffix = key_suffix
        self.store = None
        self._lock = threading.Lock()
        self._registered_buffers = set()

    def _ensure_slab_registered(self, slab: torch.Tensor) -> bool:
        """Ensure the given slab tensor is registered for zero-copy.
        Returns True on success (already registered or newly registered)."""
        if self.store is None:
            return False
        try:
            ptr = slab.data_ptr()
            if ptr in self._registered_buffers:
                return True
            size = slab.numel()  # uint8 slabs: num elements == bytes
            result = self.store.register_buffer(ptr, size)
            if result == 0:
                self._registered_buffers.add(ptr)
                return True
            else:
                logger.error(f"Failed to register buffer ptr={ptr}, size={size}, code={result}")
                return False
        except Exception as e:
            logger.error(f"Exception during buffer registration: {e}")
            return False

    @classmethod
    def from_envs(
        cls, conn_id: str, executor: Executor, **kwargs
    ) -> "MooncakeConnector":
        """Create a connector from environment variables."""
        # Do not depend on environment variable for zero-copy; hardcode to False.
        config = MooncakeConfig(
            local_hostname=envs.AIBRIX_KV_CACHE_OL_MOONCAKE_LOCAL_HOSTNAME,
            metadata_server=envs.AIBRIX_KV_CACHE_OL_MOONCAKE_METADATA_SERVER,
            global_segment_size=envs.AIBRIX_KV_CACHE_OL_MOONCAKE_GLOBAL_SEGMENT_SIZE,
            local_buffer_size=envs.AIBRIX_KV_CACHE_OL_MOONCAKE_LOCAL_BUFFER_SIZE,
            protocol=envs.AIBRIX_KV_CACHE_OL_MOONCAKE_PROTOCOL,
            rdma_devices=envs.AIBRIX_KV_CACHE_OL_MOONCAKE_RDMA_DEVICES,
            master_server_addr=envs.AIBRIX_KV_CACHE_OL_MOONCAKE_MASTER_SERVER_ADDR,
        )
        return cls(config, conn_id, executor)

    @property
    def name(self) -> str:
        return "Mooncake"

    @property
    def feature(self) -> ConnectorFeature:
        feature = ConnectorFeature()
        # Mooncake supports batch operations
        feature.mput_mget = True
        # RDMA support depends on protocol configuration
        if self.config.protocol.lower() == "rdma":
            feature.rdma = True
        return feature

    def __del__(self) -> None:
        self.close()

    def _key(self, key: bytes) -> str:
        """Convert bytes key to string key with suffix."""
        return key.hex() + self.key_suffix

    @Status.capture_exception
    def open(self) -> Status:
        """Open a connection."""
        try:
            from mooncake.store import MooncakeDistributedStore

            with self._lock:
                if self.store is None:
                    self.store = MooncakeDistributedStore()
                    result = self.store.setup(
                        self.config.local_hostname,
                        self.config.metadata_server,
                        self.config.global_segment_size,
                        self.config.local_buffer_size,
                        self.config.protocol,
                        self.config.rdma_devices,
                        self.config.master_server_addr,
                    )
                    if result != 0:
                        logger.error(f"Failed to setup Mooncake store: {result}")
                        return Status(StatusCodes.INVALID)
                    logger.info("Mooncake store opened successfully")
            return Status.ok()
        except ImportError as e:
            logger.error(f"Failed to import mooncake.store: {e}")
            return Status(StatusCodes.INVALID)
        except Exception as e:
            logger.error(f"Failed to open Mooncake store: {e}")
            return Status(StatusCodes.INVALID)

    @Status.capture_exception
    def close(self) -> Status:
        """Close a connection."""
        with self._lock:
            if self.store is not None:
                try:
                    # Unregister all buffers
                    for buffer_ptr in list(self._registered_buffers):
                        self.store.unregister_buffer(buffer_ptr)
                    self._registered_buffers.clear()

                    # Close the store
                    self.store.close()
                    self.store = None
                    logger.info("Mooncake store closed successfully")
                except Exception as e:
                    logger.error(f"Error closing Mooncake store: {e}")
                    return Status(StatusCodes.INVALID)
        return Status.ok()

    @Status.capture_exception
    def register_slabs(self, slabs: List[torch.Tensor]) -> Status:
        """Register slabs with Mooncake store for zero-copy by default."""
        if self.store is None:
            return Status(StatusCodes.INVALID)
        for slab in slabs:
            ok = self._ensure_slab_registered(slab)
            if not ok:
                return Status(StatusCodes.INVALID)
        return Status.ok()

    @Status.capture_exception
    async def exists(self, key: bytes) -> Status:
        """Check if key is in the store."""
        if self.store is None:
            return Status(StatusCodes.INVALID)

        try:
            result = await self.event_loop.run_in_executor(
                self._executor, self._sync_exists, key
            )
            return result
        except Exception as e:
            logger.error(f"Error checking key existence: {e}")
            return Status(StatusCodes.INVALID)

    def _sync_exists(self, key: bytes) -> Status:
        """Synchronous exists check."""
        try:
            str_key = self._key(key)
            exists = self.store.is_exist(str_key)
            if exists == 1:
                return Status.ok()
            elif exists == 0:
                return Status(StatusCodes.NOT_FOUND)
            else:
                return Status(StatusCodes.INVALID)
        except Exception as e:
            logger.error(f"Error in sync exists: {e}")
            return Status(StatusCodes.INVALID)

    def get_batches(
        self,
        keys: Sequence[Any],
        mrs: Sequence[MemoryRegion | Sequence[MemoryRegion]],
        batch_size: int,
    ) -> Sequence[Sequence[Tuple[bytes, MemoryRegion | Sequence[MemoryRegion]]]]:
        """Get batches for batch operations."""
        lists: List[List[Tuple[bytes, MemoryRegion | Sequence[MemoryRegion]]]] = []

        for key, mr in zip(keys, mrs):
            if len(lists) == 0 or len(lists[-1]) >= batch_size:
                lists.append([(key, mr)])
            else:
                lists[-1].append((key, mr))
        return lists

    @Status.capture_exception
    async def mget(
        self,
        keys: Sequence[bytes],
        mrs: Sequence[MemoryRegion | Sequence[MemoryRegion]],
    ) -> Sequence[Status]:
        """Batch get operation."""
        if self.store is None:
            return [Status(StatusCodes.INVALID)] * len(keys)

        try:
            result = await self.event_loop.run_in_executor(
                self._executor, self._sync_mget, keys, mrs
            )
            return result
        except Exception as e:
            logger.error(f"Error in mget: {e}")
            return [Status(StatusCodes.INVALID)] * len(keys)

    def _sync_mget(
        self,
        keys: Sequence[bytes],
        mrs: Sequence[MemoryRegion | Sequence[MemoryRegion]]
    ) -> Sequence[Status]:
        """Synchronous batch get with zero-copy by default when possible."""
        # Try zero-copy path if all mrs are single MemoryRegion and buffers are registered
        try:
            if all(not isinstance(mr, Sequence) for mr in mrs):
                # Ensure registration for all slabs
                for mr in mrs:  # type: ignore
                    if not self._ensure_slab_registered(mr.slab):  # type: ignore
                        raise RuntimeError("registration_failed")
                str_keys = [self._key(key) for key in keys]
                ptrs = [mr.data_ptr() for mr in mrs]  # type: ignore
                sizes = [len(mr) for mr in mrs]  # type: ignore
                results = self.store.batch_get_into(str_keys, ptrs, sizes)
                statuses: List[Status] = []
                for r in results:
                    if r > 0:
                        statuses.append(Status.ok())
                    elif r == 0:
                        statuses.append(Status(StatusCodes.NOT_FOUND))
                    else:
                        statuses.append(Status(StatusCodes.INVALID))
                return statuses
        except Exception as e:
            if str(e) != "registration_failed":
                logger.warning(f"Zero-copy batch get failed, fallback to regular path: {e}")
            # Fallback to regular batch operations below

        # Regular (non-zero-copy) batch path
        try:
            statuses = []
            str_keys = [self._key(key) for key in keys]
            values = self.store.get_batch(str_keys)
            for value, mr in zip(values, mrs):
                if value and len(value) > 0:
                    if isinstance(mr, Sequence):
                        chunk_size = len(value) // len(mr)
                        for j, m in enumerate(mr):
                            start_idx = j * chunk_size
                            end_idx = start_idx + chunk_size
                            m.fill(value[start_idx:end_idx])
                    else:
                        mr.fill(value)
                    statuses.append(Status.ok())
                else:
                    statuses.append(Status(StatusCodes.NOT_FOUND))
            return statuses
        except Exception as e:
            logger.error(f"Error in regular batch get: {e}")
            return [Status(StatusCodes.INVALID)] * len(keys)

    @Status.capture_exception
    async def mput(
        self,
        keys: Sequence[bytes],
        mrs: Sequence[MemoryRegion | Sequence[MemoryRegion]],
    ) -> Sequence[Status]:
        """Batch put operation."""
        if self.store is None:
            return [Status(StatusCodes.INVALID)] * len(keys)

        try:
            result = await self.event_loop.run_in_executor(
                self._executor, self._sync_mput, keys, mrs
            )
            return result
        except Exception as e:
            logger.error(f"Error in mput: {e}")
            return [Status(StatusCodes.INVALID)] * len(keys)

    def _sync_mput(
        self,
        keys: Sequence[bytes],
        mrs: Sequence[MemoryRegion | Sequence[MemoryRegion]]
    ) -> Sequence[Status]:
        """Synchronous batch put with zero-copy by default when possible."""
        # Try zero-copy path if all mrs are single MemoryRegion and buffers are registered
        try:
            if all(not isinstance(mr, Sequence) for mr in mrs):
                for mr in mrs:  # type: ignore
                    if not self._ensure_slab_registered(mr.slab):  # type: ignore
                        raise RuntimeError("registration_failed")
                str_keys = [self._key(key) for key in keys]
                ptrs = [mr.data_ptr() for mr in mrs]  # type: ignore
                sizes = [len(mr) for mr in mrs]  # type: ignore
                results = self.store.batch_put_from(str_keys, ptrs, sizes)
                statuses: List[Status] = []
                for r in results:
                    if r == 0:
                        statuses.append(Status.ok())
                    else:
                        statuses.append(Status(StatusCodes.INVALID))
                return statuses
        except Exception as e:
            if str(e) != "registration_failed":
                logger.warning(f"Zero-copy batch put failed, fallback to regular path: {e}")
            # Fallback to regular batch operations below

        # Regular (non-zero-copy) batch path
        try:
            str_keys = [self._key(key) for key in keys]
            values = []
            for mr in mrs:
                if isinstance(mr, Sequence):
                    value = b"".join(m.tobytes() for m in mr)
                else:
                    value = mr.tobytes()
                values.append(value)
            result = self.store.put_batch(str_keys, values)
            if result == 0:
                return [Status.ok()] * len(keys)
            else:
                return [Status(StatusCodes.INVALID)] * len(keys)
        except Exception as e:
            logger.error(f"Error in regular batch put: {e}")
            return [Status(StatusCodes.INVALID)] * len(keys)

    @Status.capture_exception
    async def get(
        self, key: bytes, mr: MemoryRegion | Sequence[MemoryRegion]
    ) -> Status:
        """Get a single value."""
        if self.store is None:
            return Status(StatusCodes.INVALID)

        try:
            result = await self.event_loop.run_in_executor(
                self._executor, self._sync_get, key, mr
            )
            return result
        except Exception as e:
            logger.error(f"Error in get: {e}")
            return Status(StatusCodes.INVALID)

    def _sync_get(self, key: bytes, mr: MemoryRegion | Sequence[MemoryRegion]) -> Status:
        """Synchronous get operation, zero-copy by default when possible."""
        try:
            str_key = self._key(key)
            # Zero-copy path for single MR
            if not isinstance(mr, Sequence):
                if not self._ensure_slab_registered(mr.slab):
                    raise RuntimeError("registration_failed")
                bytes_read = self.store.get_into(str_key, mr.data_ptr(), len(mr))
                if bytes_read > 0:
                    return Status.ok()
                elif bytes_read == 0:
                    return Status(StatusCodes.NOT_FOUND)
                else:
                    return Status(StatusCodes.INVALID)
        except Exception as e:
            if str(e) != "registration_failed":
                logger.warning(f"Zero-copy get failed, fallback to regular path: {e}")
            # Fallback to regular path below
        try:
            # Regular operation
            value = self.store.get(str_key)
            if value and len(value) > 0:
                if isinstance(mr, Sequence):
                    chunk_size = len(value) // len(mr)
                    for i, m in enumerate(mr):
                        start_idx = i * chunk_size
                        end_idx = start_idx + chunk_size
                        m.fill(value[start_idx:end_idx])
                else:
                    mr.fill(value)
                return Status.ok()
            else:
                return Status(StatusCodes.NOT_FOUND)
        except Exception as e:
            logger.error(f"Error in sync get: {e}")
            return Status(StatusCodes.INVALID)

    @Status.capture_exception
    async def put(
        self, key: bytes, mr: MemoryRegion | Sequence[MemoryRegion]
    ) -> Status:
        """Put a single key-value pair."""
        if self.store is None:
            return Status(StatusCodes.INVALID)

        try:
            result = await self.event_loop.run_in_executor(
                self._executor, self._sync_put, key, mr
            )
            return result
        except Exception as e:
            logger.error(f"Error in put: {e}")
            return Status(StatusCodes.INVALID)

    def _sync_put(self, key: bytes, mr: MemoryRegion | Sequence[MemoryRegion]) -> Status:
        """Synchronous put operation, zero-copy by default when possible."""
        try:
            str_key = self._key(key)
            # Zero-copy path for single MR
            if not isinstance(mr, Sequence):
                if not self._ensure_slab_registered(mr.slab):
                    raise RuntimeError("registration_failed")
                result = self.store.put_from(str_key, mr.data_ptr(), len(mr))
                if result == 0:
                    return Status.ok()
                else:
                    return Status(StatusCodes.INVALID)
        except Exception as e:
            if str(e) != "registration_failed":
                logger.warning(f"Zero-copy put failed, fallback to regular path: {e}")
            # Fallback to regular path below
        try:
            # Regular operation
            if isinstance(mr, Sequence):
                value = b"".join(m.tobytes() for m in mr)
            else:
                value = mr.tobytes()
            result = self.store.put(str_key, value)
            if result == 0:
                return Status.ok()
            else:
                return Status(StatusCodes.INVALID)
        except Exception as e:
            logger.error(f"Error in sync put: {e}")
            return Status(StatusCodes.INVALID)

    @Status.capture_exception
    async def delete(self, key: bytes) -> Status:
        """Delete a key."""
        if self.store is None:
            return Status(StatusCodes.INVALID)

        try:
            result = await self.event_loop.run_in_executor(
                self._executor, self._sync_delete, key
            )
            return result
        except Exception as e:
            logger.error(f"Error in delete: {e}")
            return Status(StatusCodes.INVALID)

    def _sync_delete(self, key: bytes) -> Status:
        """Synchronous delete operation.
        Some Mooncake builds return the number of removed objects (>=0) on success.
        Treat any non-negative result as success; negatives as error.
        """
        try:
            str_key = self._key(key)
            # Some deployments may reject deletion while RDMA buffers are registered.
            # Proactively unregister all known buffers before delete.
            try:
                for buffer_ptr in list(self._registered_buffers):
                    try:
                        self.store.unregister_buffer(buffer_ptr)
                    except Exception:
                        pass
                self._registered_buffers.clear()
            except Exception:
                pass

            result = self.store.remove(str_key)
            logger.info(f"remove({str_key}) -> {result}")
            if result >= 0:
                return Status.ok()

            # Handle well-known error codes per official docs
            if result == -704:  # OBJECT_NOT_FOUND
                return Status(StatusCodes.NOT_FOUND)
            if result == -706:  # OBJECT_HAS_LEASE
                logger.error("Mooncake remove blocked by lease (-706). Treating as OK per policy.")
                return Status.ok()

            return Status(StatusCodes.INVALID)
        except Exception as e:
            logger.error(f"Error in sync delete: {e}")
            return Status(StatusCodes.INVALID)
