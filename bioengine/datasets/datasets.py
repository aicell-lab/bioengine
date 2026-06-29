import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import httpx

from bioengine.datasets.chunk_cache import ChunkCache, _DEFAULT_CACHE_SIZE_GB

# Transport errors that signal the cached data-server URL may now point at
# the wrong endpoint (container restarted on a new IP, pod rescheduled,
# bridge re-attached). On any of these we re-discover via refresh() and
# retry the request once. 4xx/5xx HTTP status errors are deliberately NOT
# in this list — those indicate the server is reachable but unhappy with
# the request, and re-discovering wouldn't help.
_STALE_URL_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
)


class BioEngineDatasets:
    """
    Client interface for accessing remote scientific datasets in BioEngine.

    This class provides a comprehensive client interface for accessing and streaming
    scientific datasets managed by the BioEngine Datasets service. It handles the
    complete lifecycle from discovery through connection to data access, with robust
    error handling and efficient streaming for large scientific data.

    The client implementation follows an asynchronous pattern for non-blocking I/O
    operations, making it suitable for use in interactive computing environments and
    high-performance applications where responsiveness is critical. It handles secure
    authentication, efficient streaming, and integration with scientific data formats
    through the Zarr protocol.

    The BioEngineDatasets client integrates with:
    - HTTP Zarr store for efficient partial data access
    - Ray Serve deployments for model-data integration
    - Remote dataset services with authentication
    - Asynchronous operation for high-performance data access

    Key Features:
    - Efficient partial data access through HttpZarrStore
    - Service discovery and connection management
    - Secure dataset access with authentication tokens
    - Rich metadata access through manifest files
    - Asynchronous API for non-blocking data operations
    - Integration with AnnData and other scientific data formats
    - Robust error handling with meaningful error messages

    Implementation Details:
    The client uses HTTP as the transport protocol and implements the Zarr store API
    for partial data access, allowing efficient operations on large datasets without
    loading entire files into memory.

    Dataset Access Workflow:
    The class implements a list → access → close pattern for dataset interaction,
    where datasets are first discovered through listing, then accessed through
    the open_file method which returns a Zarr group, and finally connections
    are managed automatically.

    Attributes:
        service_url (str): URL for the BioEngine Datasets service
        http_client (httpx.AsyncClient): HTTP client for service communication
    """

    # Path of the per-host discovery file written by the data server.
    # Class attribute so refresh() / _resolve_service_url() use the same path.
    _CURRENT_SERVER_FILE = Path.home() / ".bioengine" / "datasets" / "bioengine_current_server"

    def __init__(
        self,
        data_server_url: Optional[str] = "auto",  # set to None for no data server
        hypha_token: Optional[str] = None,
        chunk_cache_size_gb: int = _DEFAULT_CACHE_SIZE_GB,
        logger: logging.Logger = logging.getLogger("BioEngineDatasets"),
    ):
        """
        Initialize the BioEngineDatasets client for dataset access.

        Sets up a client connection to the BioEngine Datasets service for
        accessing scientific datasets. Configures the connection with proper
        authentication and logging for tracking access patterns.

        Args:
            data_server_url: URL of the datasets server, ``"auto"`` to discover
                via the per-host file written by the data server (default),
                or ``None`` to disable remote access entirely. When ``"auto"``
                and the file does not exist yet, the URL is re-resolved
                lazily on every dataset operation, so a data server started
                after the client is constructed will be picked up automatically.
            hypha_token: Optional default authentication token for accessing protected datasets
            chunk_cache_size_gb: Size of the in-memory LRU chunk cache for zarr data in GB.
                All zarr stores opened by this client share one cache. Pass 0 to disable.

        Note:
            When data_server_url is None, only local dataset access will be available.
            The client uses a unique replica_id for tracking access patterns in logs.
        """
        self.logger = logger
        self.logger.info(f"Initializing {self.__class__.__name__}")
        self.chunk_cache = ChunkCache(max_size_gb=chunk_cache_size_gb, logger=logger)

        # Remember the user's intent verbatim. "auto" stays "auto" so we can
        # re-try the discovery file later if the data server hadn't started
        # by the time this client was constructed.
        self._requested_data_server_url: Optional[str] = data_server_url
        self.service_url: Optional[str] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.default_token = hypha_token

        # First attempt at construction. Non-fatal — a still-missing
        # discovery file just means future operations will retry.
        self._resolve_service_url(initial=True)

    def _resolve_service_url(self, *, initial: bool = False) -> None:
        """Lazily attach to the data server (no-op once attached).

        Called at construction and at the top of every public method that
        uses ``service_url``. The cheap path — ``service_url`` already set —
        is a single attribute check; the slow path reads one small file.

        Args:
            initial: True when called from ``__init__``. Controls whether the
                "discovery file not found" warning is emitted (we want one
                warning at startup, not a fresh warning on every operation
                for the lifetime of a client that never sees a data server).
        """
        if self.service_url is not None:
            return  # already attached

        requested = self._requested_data_server_url
        if requested is None:
            return  # caller opted out of remote datasets

        if requested == "auto":
            try:
                requested = self._CURRENT_SERVER_FILE.read_text().strip()
            except FileNotFoundError:
                if initial:
                    self.logger.warning(
                        f"No current data server found at "
                        f"'{self._CURRENT_SERVER_FILE}'. "
                        f"Will retry on each dataset operation; "
                        f"datasets will become available once the data server starts."
                    )
                return  # leave service_url None; next operation retries

        self.service_url = requested.rstrip("/")
        self.http_client = httpx.AsyncClient(timeout=20)  # seconds
        self.logger.info(f"Data server URL: {self.service_url}")

    async def refresh(self) -> Optional[str]:
        """Force re-discovery of the data server URL.

        Tears down the current HTTP client and re-runs the resolution that
        ``__init__`` did. Useful when the data server has restarted on a
        different URL, or when a caller wants to force a re-check without
        waiting for the next dataset operation to do it lazily.

        Returns:
            The newly-resolved service URL, or ``None`` if the data server
            is still unavailable. Callers can use the return value to decide
            whether to proceed or wait longer.
        """
        if self.http_client is not None:
            try:
                await self.http_client.aclose()
            except Exception as e:
                self.logger.debug(f"Closing previous http_client raised: {e}")
        self.service_url = None
        self.http_client = None
        self._resolve_service_url(initial=False)
        return self.service_url

    async def _with_url_recovery(
        self, op: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Run ``op``; on transport error, refresh the URL and retry once.

        Catches the connection/read/timeout shapes that signal the cached
        data-server URL has gone stale (data-server container restarted on
        a new IP, pod rescheduled, etc.) and re-runs ``op`` once after
        ``refresh()`` updates ``self.service_url`` from the discovery file.

        ``op`` is a zero-arg async callable that must reference
        ``self.service_url`` at *call time* (not bake it into a literal
        before invocation), so the retry picks up the refreshed value.
        4xx/5xx HTTP errors are re-raised without a retry — those mean the
        server is reachable but unhappy, and re-discovery wouldn't help.
        """
        try:
            return await op()
        except _STALE_URL_TRANSPORT_ERRORS as exc:
            self.logger.warning(
                f"Data server transport error against '{self.service_url}' "
                f"({type(exc).__name__}: {exc}); forcing re-discovery and "
                f"retrying once."
            )
            new_url = await self.refresh()
            if new_url is None:
                # No new URL available; nothing more we can do.
                raise
            self.logger.info(
                f"Data server re-discovered at '{new_url}'; retrying."
            )
            return await op()

    async def set_chunk_cache_size_gb(self, gb: int) -> None:
        """
        Change the chunk cache size limit at runtime.

        Immediately evicts the least-recently-used chunks if the current cache
        usage exceeds the new limit.

        Args:
            gb: New cache size in GB. Pass 0 to effectively disable caching.
        """
        await self.chunk_cache.resize(gb)

    async def ping_data_server(self):
        """
        Verify connectivity to the dataset service.

        Tests the connection to the dataset service by sending a simple ping request.
        This method should be called before attempting to access datasets to ensure
        that the service is available and responsive.

        Raises:
            RuntimeError: If the connection to the data server fails for any reason
        """
        self._resolve_service_url()
        if self.service_url is None:
            return

        from bioengine.datasets.utils import get_url_with_retry

        async def _ping() -> None:
            await get_url_with_retry(
                url=f"{self.service_url}/ping",
                raise_for_status=True,
                http_client=httpx.AsyncClient(timeout=3.0),  # short timeout for ping
                logger=self.logger,
            )

        try:
            await self._with_url_recovery(_ping)
        except Exception as e:
            self.logger.error(f"Connection to data server failed: {e}")
            raise RuntimeError("Connection to data server failed")

    async def list_datasets(self) -> Dict[str, dict]:
        """
        Retrieve a dictionary of available datasets from the service.

        Queries the dataset service for all datasets that are available to the current
        user. This is typically the first step in the dataset access workflow and
        provides the names needed for subsequent operations.

        Returns:
            Dictionary of dataset names and their manifest available to the current user.
            Empty dictionary if service_url is None.

        Raises:
            httpx.HTTPStatusError: If the request fails due to HTTP error
        """
        self._resolve_service_url()
        if self.service_url is None:
            return {}

        from bioengine.datasets.utils import get_url_with_retry

        start_time = asyncio.get_event_loop().time()

        async def _list():
            return await get_url_with_retry(
                url=f"{self.service_url}/datasets",
                raise_for_status=True,
                http_client=self.http_client,
                logger=self.logger,
            )

        response = await self._with_url_recovery(_list)
        datasets = response.json()
        end_time = asyncio.get_event_loop().time()
        self.logger.debug(
            f"Listed {len(datasets)} dataset(s) "
            f"in {end_time - start_time:.2f} seconds"
        )

        return datasets

    async def list_files(
        self,
        dataset_id: str,
        dir_path: Optional[str] = None,
        token: Optional[str] = None,
    ) -> List[str]:
        """
        Retrieve a list of available files within a specific dataset.

        Queries the dataset service for all files available within the specified dataset.
        This is typically the second step in the dataset access workflow, after listing
        available datasets. The method handles authentication and access control through
        the optional token parameter.

        Args:
            dataset_id: Name of the dataset to list files from
            token: Optional authentication token for accessing protected datasets

        Returns:
            List of file paths available within the specified dataset

        Raises:
            ValueError: If the service_url is None or dataset does not exist
            httpx.HTTPStatusError: If the request fails due to HTTP error
        """
        self._resolve_service_url()
        if self.service_url is None:
            raise ValueError(
                f"Dataset '{dataset_id}' could not be accessed. No connection to data server."
            )

        from bioengine.datasets.utils import get_url_with_retry

        start_time = asyncio.get_event_loop().time()
        token = token or self.default_token

        params = {}
        if dir_path is not None:
            params["dir_path"] = dir_path
        if token is not None:
            params["token"] = token

        async def _list_files():
            return await get_url_with_retry(
                url=f"{self.service_url}/datasets/{dataset_id}/files",
                params=params,
                raise_for_status=True,
                http_client=self.http_client,
                logger=self.logger,
            )

        response = await self._with_url_recovery(_list_files)
        files = response.json()
        end_time = asyncio.get_event_loop().time()
        self.logger.debug(
            f"Listed {len(files)} file(s) in dataset "
            f"'{dataset_id}' in {end_time - start_time:.2f} seconds"
        )

        return files

    async def get_file(
        self,
        dataset_id: str,
        file_path: str,
        token: Optional[str] = None,
    ) -> Union["HttpZarrStore", bytes]:
        """
        Access a remote data file as a streamable Zarr store for efficient data operations.

        This is the primary method for accessing dataset content, providing access to
        the data through Zarr's efficient partial data access mechanisms. The method
        validates dataset and file existence, handles authentication, and returns a
        connected Zarr group for immediate data access.

        Dataset Access Process:
        1. Validates dataset existence through list_datasets
        2. Checks file availability through list_files
        3. Creates and returns an HttpZarrStore for efficient streaming access

        Args:
            dataset_id: Name of the dataset to access
            file_path: Specific file path within the dataset to access
            token: Optional authentication token for accessing protected datasets

        Returns:
            Connected HttpZarrStore instance for the specified dataset file

        Raises:
            ValueError: If dataset/file doesn't exist
            RuntimeError: If connection to data server fails
        """
        start_time = asyncio.get_event_loop().time()
        token = token or self.default_token

        available_datasets = await self.list_datasets()
        if dataset_id not in available_datasets.keys():
            raise ValueError(f"Dataset '{dataset_id}' does not exist")

        _file_path = Path(file_path)

        if _file_path.suffix == ".zarr":
            # zarr stores are directories — check that at least one file exists under them
            lookup_dir = _file_path.as_posix()
            available_files = await self.list_files(
                dataset_id=dataset_id, dir_path=lookup_dir, token=token
            )
            if not available_files:
                raise ValueError(
                    f"Zarr store '{_file_path.as_posix()}' not found in dataset '{dataset_id}'"
                )
        else:
            lookup_dir = (
                _file_path.parent.as_posix() if _file_path.parent != Path(".") else None
            )
            available_files = await self.list_files(
                dataset_id=dataset_id, dir_path=lookup_dir, token=token
            )
            if _file_path.as_posix() not in available_files and _file_path.name not in available_files:
                raise ValueError(
                    f"File '{_file_path.as_posix()}' not found in dataset '{dataset_id}'"
                )

        if _file_path.suffix == ".zarr":
            try:
                from bioengine.datasets.http_zarr_store import HttpZarrStore
            except ImportError as e:
                raise ImportError("Unable to load HttpZarrStore") from e

            zarr_path = _file_path.as_posix().lstrip("/")
            file_output = HttpZarrStore(
                base_url=f"{self.service_url}/data/{dataset_id}/{zarr_path}",
                token=token,
                chunk_cache=self.chunk_cache,
                logger=self.logger,
                dataset_id=dataset_id,
                zarr_path=zarr_path,
            )
        else:
            from bioengine.datasets.utils import get_url_with_retry

            params = {"token": token} if token else {}

            async def _get():
                return await get_url_with_retry(
                    url=f"{self.service_url}/data/{dataset_id}/{_file_path.as_posix()}",
                    params=params,
                    raise_for_status=True,
                    http_client=self.http_client,
                    logger=self.logger,
                )

            response = await self._with_url_recovery(_get)
            file_output = response.content

        end_time = asyncio.get_event_loop().time()
        self.logger.debug(
            f"Time taken to get file '{_file_path.as_posix()}' from dataset "
            f"'{dataset_id}': {end_time - start_time:.2f} seconds"
        )

        return file_output

    def open_remote_zarr(
        self,
        uri: str,
        token: Optional[str] = None,
    ) -> "HttpZarrStore":
        """
        Stream an OME-Zarr (or any Zarr) from an arbitrary HTTPS URI.

        Bypasses the local data server entirely — callers pass a URI obtained
        elsewhere (e.g. from the BioImage Archive search API) and get back a
        Zarr store that streams chunks on demand via HTTP byte-range requests.
        The store reuses this client's shared chunk cache.

        Args:
            uri: HTTPS URL of the Zarr root (the directory containing
                ``.zarray`` for a Zarr v2 array, ``.zattrs`` for a Zarr v2
                group, or ``zarr.json`` for a Zarr v3 store).
            token: Optional auth token, appended as ``?token=`` to chunk
                URLs. Public Zarr roots (e.g. BioImage Archive) don't need it.

        Returns:
            HttpZarrStore that can be opened with ``zarr.open(store, ...)``.

        Example:
            ```python
            # Search the BioImage Archive for an OME-Zarr image
            import httpx
            r = httpx.get(
                "https://beta.bioimagearchive.org/search/v1/search/fts/image",
                params={"q": "cellpose", "pageSize": 1},
            ).json()
            hit = r["hits"]["hits"][0]["_source"]
            uri = next(
                rep["file_uri"][0]
                for rep in hit["representation"]
                if rep["image_format"] == ".ome.zarr"
            )

            # Stream the OME-Zarr
            import zarr
            store = datasets.open_remote_zarr(uri)
            root = zarr.open(store, mode="r")
            ```
        """
        try:
            from bioengine.datasets.http_zarr_store import HttpZarrStore
        except ImportError as e:
            raise ImportError("Unable to load HttpZarrStore") from e

        self.logger.debug(f"Opening remote Zarr at {uri}")
        return HttpZarrStore(
            base_url=uri,
            token=token,
            chunk_cache=self.chunk_cache,
            logger=self.logger,
        )

    async def save_file(
        self,
        filename: str,
        content: Union[bytes, str],
        public: bool = False,
        token: Optional[str] = None,
    ) -> dict:
        """
        Save a file to the datasets server.

        Public files go to saved/public/ and cannot be overwritten once created.
        Private files go to saved/{user_id}/ and allow overwriting. The user
        identity is derived from the Hypha token (GitHub-backed OAuth).

        Args:
            filename: Name of the file (no path separators).
            content: File content as bytes or str (str is encoded to UTF-8).
            public: True → world-readable, no overwrite. False (default) → owner-only, overwrite allowed.
            token: Hypha authentication token. Falls back to the default token.

        Returns:
            Dict with dataset_id, filename, size, and public flag.
        """
        self._resolve_service_url()
        if self.service_url is None:
            raise ValueError("No connection to data server.")

        token = token or self.default_token
        if isinstance(content, str):
            content = content.encode("utf-8")

        params = {"filename": filename, "public": str(public).lower()}
        if token:
            params["token"] = token

        response = await self.http_client.post(
            url=f"{self.service_url}/save",
            params=params,
            content=content,
        )
        response.raise_for_status()
        result = response.json()
        self.logger.debug(
            f"Saved '{filename}' to '{result['dataset_id']}' ({result['size']} bytes)"
        )
        return result

    async def list_saved_files(
        self,
        public: bool = False,
        dir_path: Optional[str] = None,
        token: Optional[str] = None,
    ) -> List[str]:
        """
        List files in the public or private saved directory.

        Args:
            public: True to list the public directory. False (default) lists the
                caller's private directory.
            dir_path: Optional subdirectory within the save directory to list.
            token: Hypha authentication token. Falls back to the default token.
                Required when public=False.

        Returns:
            List of file paths relative to the save directory root.
        """
        self._resolve_service_url()
        if self.service_url is None:
            raise ValueError("No connection to data server.")

        from bioengine.datasets.utils import get_url_with_retry

        token = token or self.default_token
        params: dict = {"public": str(public).lower()}
        if dir_path is not None:
            params["dir_path"] = dir_path
        if token:
            params["token"] = token

        response = await get_url_with_retry(
            url=f"{self.service_url}/saved",
            params=params,
            raise_for_status=True,
            http_client=self.http_client,
            logger=self.logger,
        )
        return response.json()

    async def get_saved_file(
        self,
        filename: str,
        public: bool = False,
        token: Optional[str] = None,
    ) -> bytes:
        """
        Retrieve a previously saved file.

        Routes to the public or private directory based on the public flag and
        the user identity derived from the token.

        Args:
            filename: Name of the file (may contain slashes for subdirectories).
            public: True to fetch from the public directory. False (default) fetches
                from the caller's private directory.
            token: Hypha authentication token. Falls back to the default token.
                Required when public=False.

        Returns:
            Raw file content as bytes.
        """
        self._resolve_service_url()
        if self.service_url is None:
            raise ValueError("No connection to data server.")

        from bioengine.datasets.utils import get_url_with_retry

        token = token or self.default_token
        params: dict = {"public": str(public).lower()}
        if token:
            params["token"] = token

        response = await get_url_with_retry(
            url=f"{self.service_url}/saved/{filename}",
            params=params,
            raise_for_status=True,
            http_client=self.http_client,
            logger=self.logger,
        )
        return response.content


if __name__ == "__main__":
    """
    Requires the following packages:

    ```
    pip install -r requirements-datasets.txt
    pip install "anndata[lazy]==0.12.2" # for read_lazy
    ```
    """
    import os
    from pathlib import Path

    from anndata.experimental import read_lazy

    async def test_bioengine_datasets():
        bioengine_datasets = BioEngineDatasets(
            data_server_url="auto",
            hypha_token=os.environ["HYPHA_TOKEN"],
        )
        await bioengine_datasets.ping_data_server()

        available_datasets = await bioengine_datasets.list_datasets()

        dataset_id = list(available_datasets.keys())[0]

        available_files = await bioengine_datasets.list_files(dataset_id)
        print(available_files)

        zarr_file = [f for f in available_files if f.endswith(".zarr")][0]

        store = await bioengine_datasets.get_file(
            dataset_id=dataset_id, file_path=zarr_file
        )
        print(store)

        file_content = await bioengine_datasets.get_file(
            dataset_id=dataset_id, file_path="filter_129.zarr/zarr.json"
        )

        # Load it as lazy AnnData object
        adata = await asyncio.to_thread(read_lazy, store, load_annotation_index=True)
        print(adata)
        print(adata.obs)
        print(adata.X)

        # Load a slice of data
        adata.layers["X_binned"][1, :].compute()
        # loaded chunks: 0.0, 0.1, 0.2, 0.4

        # Test presigned URL caching
        adata.layers["X_binned"][1, :].compute()
        # -> does not need to request new presigned URL
        # -> still needs to fetch same data chunks again

        # Load next slice of data
        adata.layers["X_binned"][2, :].compute()
        # -> does not need to request new presigned URL - slice in in same chunks
        # -> needs to fetch new data chunks

        adata.layers["X_binned"][:, 1].compute()
        # loaded chunks: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
        # -> needs to request new presigned URL - different chunks, one overlap

        # Cleanup
        store.close()

    asyncio.run(test_bioengine_datasets())
