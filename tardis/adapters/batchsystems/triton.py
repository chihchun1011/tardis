from ...configuration.configuration import Configuration
from ...interfaces.batchsystemadapter import BatchSystemAdapter, MachineStatus
from ...utilities.asynccachemap import AsyncCacheMap
from ...utilities.executors.shellexecutor import ShellExecutor

import hashlib
import json
import logging
import os
import asyncio
import tempfile
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
          discovery_file: /srv/triton-discovery/endpoints.json  # optional
          options:
            pool: c03-001-151.gridka.de

    discovery_file is optional. When set, every host-cache refresh also
    writes the current drone_uuid -> host/port mapping to this path as
    JSON, atomically (temp file + rename), so external consumers such as
    Prometheus (file_sd_config) or Envoy (filesystem-based EDS) can find
    live Triton endpoints without talking to condor_q or TARDIS directly.
    Leave unset to skip this (no discovery file is written).

    triton_http_port/triton_metrics_port are BASE ports, not literal ones.
    Multiple pilots can land on the same GPU node, so each drone's actual
    ports are base + a stable per-drone offset derived from its
    TardisDroneUuid (see _port_offset). start_pilot.sh derives the same
    offset from the same UUID to pick tritonserver's --http-port/
    --grpc-port/--metrics-port, so the two sides always agree without any
    extra coordination.
    """

    PORT_OFFSET_RANGE = 10000

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

        # Optional: path to export the live drone_uuid -> host/port mapping.
        # Unset by default so existing deployments see no behaviour change.
        self._discovery_file = getattr(config.BatchSystem, "discovery_file", None)
        if self._discovery_file is not None and not os.path.dirname(self._discovery_file):
            # A bare filename (no directory component) would otherwise write
            # silently into this process's current working directory.
            logger.warning(
                f"discovery_file={self._discovery_file!r} has no directory component - "
                f"it will be written to this process's current working directory "
                f"({os.getcwd()}) on every refresh. Use a path with an explicit "
                f"directory (e.g. '/srv/triton-discovery/{self._discovery_file}') instead."
            )

        # Testing-only escape hatch: if set, get_utilisation() reads a float
        # from this file instead of querying Triton's real /metrics, so
        # scale-up/scale-down can be driven by hand (e.g. `echo 0.95 > file`)
        # without needing real inference traffic. Leave unset for production.
        self._simulated_utilisation_file = getattr(
            config.BatchSystem, "simulated_utilisation_file", None
        )

        # Cache: drone_uuid → hostname string
        # Re-queried at most every max_age minutes
        self._host_cache = AsyncCacheMap(
            update_coroutine=self._update_host_cache,
            max_age=max_age * 60,
        )

        # drone_uuids that drain_machine() has been called for. Triton has
        # no graceful-drain protocol to wait on, so once a drone lands here
        # get_machine_status() reports it Drained immediately - otherwise
        # it would keep reporting Available forever (Triton stays healthy)
        # and DrainingState would bounce back to DrainState in an infinite
        # loop, never reaching DisintegrateState/CleanupState/condor_rm.
        self._draining: set[str] = set()

    async def _update_host_cache(self) -> dict:
        """
        This is the update function for the AsyncCacheMap above.
         Map drone_uuid → hostname via condor_q

        Query condor_q once for every currently-running Triton pilot job,
        rebuilding the full {drone_uuid: hostname} map from scratch - not a
        one-off snapshot. This is the update_coroutine of an AsyncCacheMap,
        so it re-runs on demand (throttled by max_age) whenever any drone
        calls _get_host().

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
            # VAR=VALUE pairs) - RemoteHost is always the last token:
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
            offset = self._port_offset(drone_uuid)
            logger.debug(
                f"Mapped drone {drone_uuid} → {hostname} "
                f"(http={self._http_port + offset}, metrics={self._metrics_port + offset})"
            )

        logger.debug(f"Host cache updated: {len(host_map)} entries")
        self._export_discovery_file(host_map)
        return host_map

    def _export_discovery_file(self, host_map: dict[str, str]) -> None:
        """
        Write the current drone_uuid → host/port mapping to self._discovery_file
        as JSON for external consumers. No-op if discovery_file isn't configured.

        Two files are written: prometheus_targets.json and endpoints.json for scripts/Envoy.

        Both are written atomically (temp file + os.replace) so a reader
        never sees a half-written file mid-update - file_sd/EDS watchers
        poll for changes at arbitrary times, including while this write is
        in progress.
        """
        if self._discovery_file is None:
            return

        entries = [
            {
                "drone_uuid": drone_uuid,
                "host": host,
                "http_port": self._http_port + self._port_offset(drone_uuid),
                "metrics_port": self._metrics_port + self._port_offset(drone_uuid),
            }
            for drone_uuid, host in host_map.items()
        ]
        self._atomic_write_json(self._discovery_file, entries)

        prometheus_targets = [
            {
                "targets": [f"{entry['host']}:{entry['metrics_port']}"],
                "labels": {"drone_uuid": entry["drone_uuid"]},
            }
            for entry in entries
        ]
        prometheus_file = os.path.join(
            os.path.dirname(self._discovery_file) or ".", "prometheus_targets.json"
        )
        self._atomic_write_json(prometheus_file, prometheus_targets)

    @staticmethod
    def _atomic_write_json(path: str, data) -> None:
        directory = os.path.dirname(path) or "."
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=directory, prefix=".triton_discovery_", suffix=".tmp"
            )
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(data, f, indent=2)
            # mkstemp creates the file mode 0600 (owner-only) by default -
            # too restrictive for external readers (e.g. Prometheus, running
            # as a different uid in its own container) to open it at all.
            os.chmod(tmp_path, 0o644)
            os.replace(tmp_path, path)
        except OSError as e:
            logger.warning(f"Could not write {path}: {e}")
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    async def _get_host(self, drone_uuid: str) -> str | None:
        """
        Return the hostname for a given drone_uuid, or None if not found.
        Uses the cached host map — only re-queries condor_q when cache expires.
        """
        await self._host_cache.update_status()
        return self._host_cache.get(drone_uuid)

    @classmethod
    def _port_offset(cls, drone_uuid: str) -> int:
        """
        Stable per-drone port offset derived from drone_uuid, so multiple
        Triton pilots landing on the same node don't collide on the same
        ports. Must match the offset start_pilot.sh derives for
        --http-port/--grpc-port/--metrics-port.
        """
        digest = hashlib.md5(drone_uuid.encode()).hexdigest()[:8]
        return int(digest, 16) % cls.PORT_OFFSET_RANGE

    # ------------------------------------------------------------------
    # BatchSystemAdapter interface implementation
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
        if self._simulated_utilisation_file is not None:
            try:
                with open(self._simulated_utilisation_file) as f:
                    simulated = float(f.read().strip())
                logger.debug(f"Using simulated utilisation {simulated} for {drone_uuid}")
                return simulated
            except (OSError, ValueError) as e:
                logger.debug(
                    f"Could not read simulated utilisation from "
                    f"{self._simulated_utilisation_file}: {e}, falling back to real metrics"
                )

        host = await self._get_host(drone_uuid)
        if host is None:
            logger.debug(f"No host found for {drone_uuid}, returning utilisation 0.0")
            return 0.0

        url = f"http://{host}:{self._metrics_port + self._port_offset(drone_uuid)}/metrics"
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

        url = f"http://{host}:{self._http_port + self._port_offset(drone_uuid)}/v2/health/ready"
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
          already draining          →  Drained (nothing to wait for)
          /v2/health/ready == 200   →  Available
          anything else             →  NotAvailable
        """
        if drone_uuid in self._draining:
            return MachineStatus.Drained

        allocation = await self.get_allocation(drone_uuid)
        if allocation == 1.0:
            return MachineStatus.Available
        return MachineStatus.NotAvailable

    async def integrate_machine(self, drone_uuid: str) -> None:
        """No integration step needed — Triton starts serving immediately."""
        return

    async def disintegrate_machine(self, drone_uuid: str) -> None:
        """
        No disintegration step needed — TARDIS will condor_rm the job.
        Drop the drone from _draining here: this is the last point in the
        drone's lifecycle where get_machine_status() is still consulted
        (ShutDownState/ShuttingDownState/CleanupState never call it again),
        so it's safe to stop tracking it and avoid leaking one entry per
        drained drone for the life of the process.
        """
        self._draining.discard(drone_uuid)

    async def drain_machine(self, drone_uuid: str) -> None:
        """
        Triton has no drain concept, so there's nothing to wait for - mark
        the drone as drained immediately so get_machine_status() reports
        MachineStatus.Drained and TARDIS proceeds straight to termination
        (condor_rm) on its next check.
        """
        self._draining.add(drone_uuid)
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