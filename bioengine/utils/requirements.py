import importlib.metadata as md
import re
from typing import List, Optional

split_re = re.compile(r"(==|>=|<=|~=|>|<)")
name_sep_re = re.compile(r"[-_.]+")

# TODO: Use lock files instead of modified version ranges


def requirement_name(requirement: str) -> str:
    """
    Reduce a requirement string to its PEP 503 normalized distribution name.

    Strips the version specifier and any extras, so the name that
    ``bioengine``'s metadata declares (``httpx[http2]>=0.28.1``) compares
    equal to the plain name a caller selects by (``httpx``).

    Args:
        requirement: A pip requirement string (e.g. "httpx[http2]>=0.28.1")

    Returns:
        The normalized name (e.g. "httpx")
    """
    name = split_re.split(requirement, maxsplit=1)[0]
    return name_sep_re.sub("-", name.split("[", 1)[0].strip()).lower()


def normalize_requirement(requirement: str) -> str:
    """
    Pin a requirement to the version this process actually has installed.

    Everything this module emits ends up in an app's ``runtime_env.pip``,
    which Ray resolves against PyPI on every environment build. Reading the
    pin off the *specifier* rather than off the installed distribution is
    what let a worker carrying one ``hypha-rpc`` hand its apps another: a
    ``>=`` floor in the metadata was rewritten to ``==<floor>`` while the
    image had resolved that same floor to something newer. The version the
    worker imports is the one that has to cross the cloudpickle boundary,
    so that is the version to pin.

    Falls back to collapsing the specifier's lower bound when the
    distribution is not importable here (``>=``/``<=``/``~=`` → ``==``).
    ~= is PEP 440's compatible-release operator: ``pkg~=2.12.0`` is
    equivalent to ``>=2.12.0, <2.13``.

    Args:
        requirement: A pip requirement string (e.g., "numpy>=1.21.0",
            "pydantic~=2.12.0")

    Returns:
        Requirement pinned with == (e.g., "numpy==1.26.4",
        "httpx[http2]==0.28.1")
    """
    if not requirement:
        return requirement

    name = split_re.split(requirement, maxsplit=1)[0].strip()
    base = name.split("[", 1)[0].strip()
    if base:
        try:
            return f"{name}=={md.version(base)}"
        except md.PackageNotFoundError:
            pass

    # Replace >=, <=, ~= with == for reproducibility
    requirement = requirement.replace(">=", "==")
    requirement = requirement.replace("<=", "==")
    requirement = requirement.replace("~=", "==")

    return requirement


def get_pip_requirements(
    select: Optional[List[str]] = None, extras: Optional[List[str]] = None
) -> List[str]:
    """
    Get pip requirements from the bioengine package metadata.

    Args:
        select: Optional list of requirement names to filter by
        extras: Optional list of extras to include (e.g. ['datasets', 'dev'])

    Returns:
        List of pip requirements

    Raises:
        ValueError: If a name in ``select`` matches no requirement. A
            selector is written against bioengine's own metadata, so a miss
            is a repo-level mistake — most often a package that only exists
            in an extra the caller did not ask for.
    """
    if extras is None:
        extras = []

    metadata = md.metadata("bioengine")
    requirements = []

    # Process main requirements
    for req in metadata.get_all("Requires-Dist", []):
        # Skip requirements that are only for specific extras
        if "; extra ==" in req:
            extra_name = req.split("; extra ==")[1].strip().strip("'\"")
            if extra_name not in extras:
                continue

        # Extract the requirement name and version
        req_name = req.split(";")[0].strip()
        requirements.append(req_name)

    if select is None:
        # If select is None, return all requirements except those starting with "ray"
        filtered_requirements = [
            normalize_requirement(requirement)
            for requirement in requirements
            if requirement and not requirement.startswith("ray")
        ]
    else:
        wanted = {requirement_name(name) for name in select}
        matched = set()
        filtered_requirements = []
        for requirement in requirements:
            if not requirement or requirement.startswith("ray"):
                continue
            name = requirement_name(requirement)
            if name in wanted:
                matched.add(name)
                filtered_requirements.append(normalize_requirement(requirement))

        unmatched = sorted(wanted - matched)
        if unmatched:
            raise ValueError(
                f"No bioengine requirement matches {unmatched} "
                f"(extras={sorted(extras)}). Every selected name must resolve, "
                "or the package silently never reaches the environment this "
                "list builds."
            )

    return filtered_requirements


def update_requirements(
    requirements: List[str],
    select: Optional[List[str]] = None,
    extras: Optional[List[str]] = None,
) -> List[str]:
    """
    Update the provided list of pip requirements with the missing BioEngine requirements.
    If a requirement is already present, it will not be overwritten.
    If `select` is provided, only the requirements that match the names in `select` will be added.
    If `extras` is provided, requirements from those extras will also be included.

    Args:
        requirements: List of requirements to update
        select: Optional list of requirement names to filter by
        extras: Optional list of extras to include (e.g. ['datasets', 'dev'])

    Returns:
        Updated list of requirements
    """
    bioengine_requirements = get_pip_requirements(select, extras)

    for bioengine_requirement in bioengine_requirements:
        exists = False
        for requirement in requirements:
            if requirement_name(bioengine_requirement) == requirement_name(requirement):
                exists = True
                break

        if not exists:
            requirements.append(normalize_requirement(bioengine_requirement))

    return requirements


if __name__ == "__main__":
    # Example usage
    print(get_pip_requirements())

    print(
        get_pip_requirements(
            select=["aiortc", "httpx", "hypha-rpc", "pydantic"], extras=["worker"]
        )
    )

    print(get_pip_requirements(select=["zarr"], extras=["datasets"]))

    print(
        update_requirements(
            ["numpy==1.21.0"],
            select=["httpx", "hypha-rpc", "pydantic"],
            extras=["worker"],
        )
    )
