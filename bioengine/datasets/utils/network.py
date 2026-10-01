import asyncio
import logging
from typing import Dict, Optional

import httpx

from bioengine._app.errors import DataServerError

# Enough of the body to carry a FastAPI `detail` message without pulling a
# whole HTML error page across the Ray boundary.
_BODY_EXCERPT_CHARS = 500


def _as_data_server_error(exc: httpx.HTTPStatusError) -> DataServerError:
    response = exc.response
    url = str(response.request.url)
    body = response.text[:_BODY_EXCERPT_CHARS]
    message = f"Data server returned HTTP {response.status_code} for '{url}'"
    if body:
        message = f"{message}: {body}"
    return DataServerError(message, response.status_code, url, body)


def raise_for_data_server_status(response: httpx.Response) -> None:
    """``response.raise_for_status()``, raising the picklable error instead.

    Unchained on purpose — httpx is an implementation detail of this client,
    and chaining would print its traceback at every caller.
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _as_data_server_error(exc) from None


async def get_url_with_retry(
    url: str,
    params: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    raise_for_status: bool = False,
    http_client: Optional[httpx.AsyncClient] = None,
    logger: Optional[logging.Logger] = None,
) -> httpx.Response:
    """
    Helper method to fetch a URL with retries.

    Implements a simple retry mechanism for HTTP GET requests to handle
    transient network issues. Retries the request up to 3 times with
    exponential backoff.

    Args:
        url: The URL to fetch
    Returns:
        The HTTP response object
    Raises:
        DataServerError: If the server answered non-2xx and raise_for_status
        httpx.HTTPError: If all retry attempts fail on a transport error
    """
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=20.0)  # seconds

    if logger is None:
        logger = logging.getLogger(__name__)

    max_attempts = 4
    backoff = 0.2  # backoff: 0.2s, 0.4s, 0.8s
    backoff_multiplier = 2.0

    for attempt in range(1, max_attempts + 1):
        try:
            response = await http_client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response
        except Exception as e:
            # Don't retry on 4xx client errors (except 429 Too Many Requests)
            if isinstance(e, httpx.HTTPStatusError):
                if (
                    400 <= e.response.status_code < 500
                    and e.response.status_code != 429
                ):
                    if raise_for_status:
                        raise _as_data_server_error(e) from None

                    return response

            if attempt < max_attempts:
                # Sleep with exponential backoff before retrying
                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed for URL {url}, "
                    f"params: {params}, error: {e}. Retrying in {backoff:.2f}s..."
                )
                await asyncio.sleep(backoff)
                backoff *= backoff_multiplier
            else:
                # If we get here, all retries failed due to errors (network, transport, etc.)
                logger.error(
                    f"Failed to fetch URL '{url}' after {max_attempts} attempts: {e}"
                )
                if not isinstance(e, httpx.HTTPStatusError):
                    raise e
                if raise_for_status:
                    raise _as_data_server_error(e) from None

                return response

