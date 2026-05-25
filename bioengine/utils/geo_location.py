import asyncio
import json
import logging
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


async def _get_json(
    url: str,
    params: Optional[Dict[str, str]] = None,
) -> Any:
    """Single async GET-and-parse-JSON with a 10-second timeout (stdlib only)."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"

    def _blocking() -> Any:
        with urllib.request.urlopen(url, timeout=10.0) as response:
            return json.loads(response.read())

    return await asyncio.to_thread(_blocking)


async def fetch_centroid_coordinates(
    country: str,
    region: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[float]]:
    """
    Fetch the centroid coordinates (latitude and longitude) for a given location
    from the Nominatim geocoding API.

    Args:
        country: The name of the country (required).
        region: The name of the region (e.g., state, province, city).
        logger: Optional logger instance.
    Returns:
        A dictionary with 'latitude' and 'longitude' keys (values may be None on failure).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    query_parts = []
    if region:
        query_parts.append(region)
    query_parts.append(country)
    query = ", ".join(query_parts)

    latitude = None
    longitude = None
    try:
        data = await _get_json(
            url="https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": "1"},
        )
        if data:
            latitude = float(data[0]["lat"])
            longitude = float(data[0]["lon"])
            logger.info(
                f"Coordinates fetched for '{query}': ({latitude:.4f}, {longitude:.4f})"
            )
    except Exception as e:
        logger.warning(
            f"Failed to calculate centroid coordinates for location '{query}': {e}"
        )

    return {"latitude": latitude, "longitude": longitude}


async def _fetch_from_ipwhois(logger: logging.Logger) -> Optional[Dict]:
    """Fetch geolocation from ipwho.is — no rate limit, returns lat/lon directly."""
    data = await _get_json(url="https://ipwho.is/")
    if not data.get("success"):
        raise ValueError(f"ipwho.is returned error: {data.get('message')}")
    return {
        "region": data.get("region"),
        "country_name": data.get("country"),
        "country_code": data.get("country_code"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone", {}).get("id"),
    }


async def _fetch_from_ipapi_com(logger: logging.Logger) -> Optional[Dict]:
    """Fetch geolocation from ip-api.com — 45 req/min free, returns lat/lon directly."""
    data = await _get_json(url="http://ip-api.com/json/")
    if data.get("status") != "success":
        raise ValueError(f"ip-api.com returned error: {data.get('message')}")
    return {
        "region": data.get("regionName"),
        "country_name": data.get("country"),
        "country_code": data.get("countryCode"),
        "latitude": data.get("lat"),
        "longitude": data.get("lon"),
        "timezone": data.get("timezone"),
    }


async def _fetch_from_ipapi_co(logger: logging.Logger) -> Optional[Dict]:
    """Fetch geolocation from ipapi.co — 1,000 req/day free, returns lat/lon directly."""
    data = await _get_json(url="https://ipapi.co/json/")
    if data.get("error"):
        raise ValueError(f"ipapi.co returned error: {data.get('reason')}")
    return {
        "region": data.get("region"),
        "country_name": data.get("country_name"),
        "country_code": data.get("country_code") or data.get("country"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "timezone": data.get("timezone"),
    }


async def fetch_geolocation(
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Optional[str]]:
    """
    Fetch geo location information, trying multiple providers in order until one succeeds.

    Providers tried in order:
      1. ipwho.is     — no rate limit, returns lat/lon directly
      2. ip-api.com   — 45 req/min, returns lat/lon directly
      3. ipapi.co     — 1,000 req/day, fallback of last resort

    Returns a dict with: region, country_name, country_code, latitude, longitude, timezone.
    All values are None if all providers fail.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    providers = [
        ("ipwho.is", _fetch_from_ipwhois),
        ("ip-api.com", _fetch_from_ipapi_com),
        ("ipapi.co", _fetch_from_ipapi_co),
    ]

    for name, fetch_fn in providers:
        try:
            geo_info = await fetch_fn(logger)
            logger.info(
                f"Geographic location detected via {name}: {geo_info['region']}, "
                f"{geo_info['country_name']} "
                f"(Timezone: {geo_info['timezone']})"
            )
            return geo_info
        except Exception as e:
            logger.warning(f"Geolocation provider '{name}' failed: {e}")

    logger.error("All geolocation providers failed.")
    return {
        "region": None,
        "country_name": None,
        "country_code": None,
        "latitude": None,
        "longitude": None,
        "timezone": None,
    }


if __name__ == "__main__":
    geo_info = asyncio.run(fetch_geolocation())
    print(f"Result: {geo_info}")
