from ...configuration.configuration import Configuration
from ...interfaces.batchsystemadapter import BatchSystemAdapter, MachineStatus
from ...utilities.asynccachemap import AsyncCacheMap
from ...utilities.executors.shellexecutor import ShellExecutor

import logging
import asyncio
from typing import Iterable

import aiohttp

logger = logging.getLogger("cobald.runtime.tardis.adapters.batchsystem.triton")


class TritonAdapter(BatchSystemAdapter):
    """
    TARDIS batch system adapter for Triton Inference Server pilots.

    Instead of querying a condor_startd, this adapter:
      1. Looks up the worker hostname via condor_q (using TardisDroneUuid
         injected into the job environment by TARDIS)
      2. Fetches GPU utilisation from Triton's Prometheus /metrics endpoint
      3. Checks drone availability via Triton's /v2/health/ready endpoint

    Configuration in cobald_htc.yml:

        BatchSystem:
          adapter: TritonAdapter
          max_age: 1           # cache TTL in minutes
          triton_http_port: 8000
          triton_metrics_port: 8002
          options:
            pool: c03-001-151.gridka.de
    """

    def __init__(self):
        config = Configuration()
        self._executor = getattr(config.BatchSystem, "executor", ShellExecutor())

        # Optional pool flag for condor_q, e.g. "-pool c03-001-151.gridka.de"
        try:
            pool = config.BatchSystem.options.pool
            self._pool_flag = f"-pool {pool}"
        except AttributeError:
            self._pool_flag = ""

        self._http_port = getattr(config.BatchSystem, "triton_http_port", 8000)
        self._metrics_port = getattr(config.BatchSystem, "triton_metrics_port", 8002)
        max_age = getattr(config.BatchSystem, "max_age", 1)  # minutes

        # Cache: drone_uuid → hostname string
        # Re-queried at most every max_age minutes
        self._host_cache = AsyncCacheMap(
            update_coroutine=self._update_host_cache,
            max_age=max_age * 60,
        )

    # ------------------------------------------------------------------
    # Step 3: Map drone_uuid → hostname via condor_q
    # This is the update function for the AsyncCacheMap above.
    # It runs condor_q ONCE and builds a full {drone_uuid: host} dict,
    # rather than one condor_q call per drone (which would be slow).
    # ------------------------------------------------------------------
    async def _update_host_cache(self) -> dict:
        """
        Query condor_q for ALL running Triton pilot jobs at once.
        Returns {drone_uuid: hostname} for every job that has a RemoteHost.

        TardisDroneUuid is not a job ClassAd — TARDIS only injects it as a
        "TardisDroneUuid=<uuid>" entry inside the job's Environment ClassAd
        (see HTCondorAdapter.deploy_resource / drone_environment). So instead
        of constraining/projecting on a nonexistent TardisDroneUuid ClassAd,
        we constrain on the real +IsTardisPilot ClassAd and pull it out of
        the Environment string ourselves.
        """
        cmd = (
            f"condor_q {self._pool_flag} "
            f"-constraint 'IsTardisPilot =?= true' "
            f"-af Environment RemoteHost"
            # Output format per line (Environment is itself space-separated
            # VAR=VALUE pairs, condor normalizes it to this form regardless
            # of submit-side syntax) - RemoteHost is always the last token:
            # "TardisDroneCores=3 TardisDroneDisk=40 TardisDroneMemory=16 \
            #  TardisDroneUuid=tritongpu-01fc0022f7 slot1@gpu-node-42.gridka.de"
        )
        logger.debug(f"Updating Triton host cache: {cmd}")
        try:
            result = await self._executor.run_command(cmd)
        except Exception as e:
            logger.warning(f"condor_q failed during host cache update: {e}")
            return {}

        host_map = {}
        for line in result.stdout.splitlines():
            parts = line.strip().rsplit(maxsplit=1)
            if len(parts) != 2:
                continue
            environment, remote_host = parts
            if remote_host.lower() == "undefined":
                continue

            drone_uuid = None
            for entry in environment.split():
                if entry.startswith("TardisDroneUuid="):
                    drone_uuid = entry.split("=", 1)[1]
                    break
            if drone_uuid is None:
                continue

            # "slot1@gpu-node-42.gridka.de" → "gpu-node-42.gridka.de"
            hostname = remote_host.split("@")[-1]
            host_map[drone_uuid] = hostname
            logger.debug(f"Mapped drone {drone_uuid} → {hostname}")

        logger.debug(f"Host cache updated: {len(host_map)} entries")
        return host_map

    async def _get_host(self, drone_uuid: str) -> str | None:
        """
        Return the hostname for a given drone_uuid, or None if not found.
        Uses the cached host map — only re-queries condor_q when cache expires.
        """
        await self._host_cache.update_status()
        return self._host_cache.get(drone_uuid)

    # ------------------------------------------------------------------
    # Step 2: BatchSystemAdapter interface implementation
    # ------------------------------------------------------------------

    async def get_utilisation(self, drone_uuid: str) -> float:
        """
        GPU utilisation from Triton's Prometheus /metrics endpoint.
        Returns 0.0 if the drone is unreachable (not yet ready, or gone).

        nv_gpu_utilization is a value between 0.0 and 1.0 reported by Triton.
        TARDIS uses this to decide whether to scale up or down:
          - utilisation < low_utilisation  → scale down (drone underused)
          - utilisation > high_allocation  → scale up   (drone overloaded)
        """
        host = await self._get_host(drone_uuid)
        if host is None:
            logger.debug(f"No host found for {drone_uuid}, returning utilisation 0.0")
            return 0.0

        url = f"http://{host}:{self._metrics_port}/metrics"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        return 0.0
                    text = await resp.text()
            return self._parse_gpu_utilisation(text)
        except Exception as e:
            logger.debug(f"Could not fetch Triton metrics for {drone_uuid} at {url}: {e}")
            return 0.0

    async def get_allocation(self, drone_uuid: str) -> float:
        """
        A Triton server is either running (allocation=1.0) or not (0.0).
        We check the /v2/health/ready endpoint.
        """
        host = await self._get_host(drone_uuid)
        if host is None:
            return 0.0

        url = f"http://{host}:{self._http_port}/v2/health/ready"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return 1.0 if resp.status == 200 else 0.0
        except Exception as e:
            logger.debug(f"Triton not ready for {drone_uuid} at {url}: {e}")
            return 0.0

    async def get_machine_status(self, drone_uuid: str) -> MachineStatus:
        """
        Map Triton health → TARDIS MachineStatus:
          /v2/health/ready == 200  →  Available
          anything else            →  NotAvailable
        """
        allocation = await self.get_allocation(drone_uuid)
        if allocation == 1.0:
            return MachineStatus.Available
        return MachineStatus.NotAvailable

    async def integrate_machine(self, drone_uuid: str) -> None:
        """No integration step needed — Triton starts serving immediately."""
        return

    async def disintegrate_machine(self, drone_uuid: str) -> None:
        """No disintegration step needed — TARDIS will condor_rm the job."""
        return

    async def drain_machine(self, drone_uuid: str) -> None:
        """
        Triton has no drain concept. Returning here means TARDIS will
        move directly to termination (condor_rm) when it wants to remove a drone.
        """
        return

    @property
    def machine_meta_data_translation_mapping(self):
        """
        Not used for supply calculation in this adapter,
        but required by the interface.
        """
        from tardis.utilities.attributedict import AttributeDict
        return AttributeDict(Cores=1, Memory=1, Disk=1)

    # ------------------------------------------------------------------
    # Helper: parse Prometheus text format from /metrics
    # ------------------------------------------------------------------

    @classmethod
    def _parse_gpu_utilisation(cls, metrics_text: str) -> float:
        """
        Average nv_gpu_utilization across all GPUs reported by this Triton
        instance. Returns 0.0 if no GPU utilisation samples were found.
        """
        compute_values, _ = cls._parse_gpu_metrics(metrics_text)
        if not compute_values:
            return 0.0
        return sum(compute_values) / len(compute_values)

    @staticmethod
    def _parse_gpu_metrics(metrics_text: str) -> tuple[list[float], list[float]]:
        compute_values, memory_values = [], []
        mem_used, mem_total = {}, {}

        for line in metrics_text.splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if line.startswith("nv_gpu_utilization{"):
                try:
                    compute_values.append(float(line.split()[-1]))
                except ValueError:
                    continue
            elif line.startswith("nv_gpu_memory_used_bytes{"):
                gpu_uuid = line.split("gpu_uuid=")[1].split(",")[0].split("}")[0]
                mem_used[gpu_uuid] = float(line.split()[-1])
            elif line.startswith("nv_gpu_memory_total_bytes{"):
                gpu_uuid = line.split("gpu_uuid=")[1].split(",")[0].split("}")[0]
                mem_total[gpu_uuid] = float(line.split()[-1])

        for gpu_uuid, used in mem_used.items():
            total = mem_total.get(gpu_uuid)
            if total:
                memory_values.append(used / total)

        return compute_values, memory_values