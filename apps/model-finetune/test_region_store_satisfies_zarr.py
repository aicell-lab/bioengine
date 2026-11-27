"""The region reader's store has to satisfy whatever zarr declares abstract.

WHAT HAPPENED. ``_LengthCheckedStore`` is a ``zarr.abc.store.Store`` subclass that
hands zarr the bytes for one key and checks the length. It defined eighteen
methods and not ``set_partial_values``, which zarr 3.0.8 declares abstract — so
the class could not be INSTANTIATED, and every region read failed with

    TypeError: Can't instantiate abstract class _LengthCheckedStore
               with abstract method set_partial_values

reported by this module as ``could not open '0' in <url>``: a message that points
a reader at the network rather than at the class. Measured at zarr 3.0.8, the
floor ``requirements-entry.txt`` declares: 18 of the 36 cases in the three region
test files failed, which is every region read in them.

WHY A GREEN SUITE DID NOT SEE IT. The published app image carries no zarr, so
those three files are not collected there at all and a suite run against it
reports green with the region reader entirely unexercised. Note that the
``importorskip`` below has the same shape: where zarr is absent this file SKIPS,
which is not a failure. So whatever runs the suite has to install zarr and assert
it imports, or "green" keeps meaning "not exercised" — and it has to pin a
version in 3.0.8..3.1.2, because from 3.1.3 on ``set_partial_values`` is no longer
abstract and all 36 cases pass against the unfixed class.

WHY THIS TEST IS NOT "assert set_partial_values exists". Naming the one method
that happened to be missing pins today's zarr and nothing else: the next release
that promotes another method to abstract breaks every region read again and this
file stays green. So the assertion is made against ``Store.__abstractmethods__``
itself — zarr's own declaration of what a store must provide — and it names
whatever is missing. The second arm instantiates the class, because a subset check
on names is not the same statement as "Python will let this be constructed".
"""

from __future__ import annotations

import pytest

zarr = pytest.importorskip("zarr", reason="the region reader needs zarr")

import ngff_region as nr  # noqa: E402


def _store_class():
    """The store class, built the way the reader builds it.

    The class is declared INSIDE ``_length_checked_store_class`` so the module
    imports where zarr is absent, which is why this cannot be a plain top-level
    import. The private name is used deliberately: the alternative is widening
    the module's surface for a test, and the hazard here is internal.
    """
    cls = nr._length_checked_store_class()
    assert cls is not None, "the reader exposes no store class to check"
    return cls


def test_the_store_provides_everything_zarr_declares_abstract():
    """THE arm. Against zarr's own declaration, not against a remembered name."""
    from zarr.abc.store import Store

    required = set(getattr(Store, "__abstractmethods__", ()))
    # CONTROL: an empty requirement set would make the assertion below vacuous,
    # and that is exactly what a wrong import or a future zarr that stopped using
    # ABCMeta would produce.
    assert required, "zarr declares no abstract store methods, so this proves nothing"
    assert "get" in required, (
        f"zarr's abstract store surface looks wrong ({sorted(required)}); the "
        f"subset check below would pass on a class that reads nothing"
    )

    cls = _store_class()
    # `__abstractmethods__` ON THE SUBCLASS, not `hasattr`. An abstract method is
    # still an inherited ATTRIBUTE of the subclass, so `hasattr(cls, name)` is
    # True for every method the class fails to implement — the obvious version of
    # this assertion passed against the very defect it was written for, which the
    # control run caught. What Python actually refuses instantiation on is this
    # set, so this is both the real check and the one that names the gap.
    still_abstract = sorted(getattr(cls, "__abstractmethods__", ()))
    assert not still_abstract, (
        f"the region store leaves {still_abstract} abstract, which zarr "
        f"{zarr.__version__} declares a store must provide — so it cannot be "
        f"instantiated and every region read fails with a message naming the "
        f"URL, not this class"
    )


def test_the_store_can_actually_be_constructed():
    """A name-subset check is not the statement Python enforces. ABCMeta refuses
    instantiation, and that refusal is what reached callers as an unopenable
    store, so the construction itself is asserted."""
    cls = _store_class()
    store = cls("https://example.invalid/a.zarr/")
    assert store is not None


async def test_a_write_through_the_partial_path_is_refused_not_accepted():
    """It is read-only, and the method exists only because the ABC demands it. A
    stub that silently returned would make a write look like it had landed."""
    store = _store_class()("https://example.invalid/a.zarr/")
    with pytest.raises(nr.RegionSourceError, match="read-only"):
        await store.set_partial_values([("0/0", 0, b"x")])
