"""URL-addressed download cache for inputs that have no coordinate.

:class:`~molhub.sources.blobs.BlobStore` files an artifact under the key a
manifest already identifies it by — a coordinate plus a role. Nothing is
content-addressed and molhub never computes a digest of its own, so a cache
entry needs a coordinate, not a hash.

:class:`~molhub.dataset.csv_dataset.CSVDataset` accepts an arbitrary URL. A URL
a user typed has no coordinate, therefore no manifest entry, therefore no key
:class:`BlobStore` could file it under. That is why this class exists, and it
is not a stage on the way to something else: as long as the public API accepts
raw URLs, URL-addressed caching is the honest answer for them.

Both caches share one root, so a user who sets ``$MOLHUB_HOME`` gets one
directory rather than two::

    $MOLHUB_HOME/                default ~/.cache/molhub
      files/<key>/…              BlobStore — the cross-client layout contract
      tmp/                       BlobStore — in-flight transfers
      urls/<url-id>-<filename>   this cache

``urls/`` is **Python-local and not part of the shared layout contract**: the
TypeScript client neither reads nor writes it, and nothing here may be written
into ``files/``, which both clients read.

The URL ID is SHA-256 of the complete URL text, including its query string.
It identifies a request, not downloaded content; no payload digest is invented.
The root itself comes from :class:`BlobStore`, so root resolution has one owner.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from molhub.sources import BlobStore, FileTransfer, HttpsSource, RemoteFile

__all__ = ["DownloadCache"]

_URL_SUBDIR = "urls"
_UNSAFE = re.compile(r"[^A-Za-z0-9._@-]+")


class DownloadCache:
    """Caches remote files on disk under a name derived from their URL.

    Every path this class hands out is inside ``urls/`` and named after the URL
    it came from — that is the whole of what it does. Its one caller is
    :class:`~molhub.dataset.csv_dataset.CSVDataset`, whose raw URLs have no
    coordinate and so no :class:`BlobStore` key.

    Args:
        root: Cache root, shared with :class:`BlobStore`; downloads land in its
            ``urls/`` subdirectory. Defaults to ``$MOLHUB_HOME``, then
            ``~/.cache/molhub``.
        transfer: Anything satisfying
            :class:`~molhub.sources.source.FileTransfer`. A URL is already
            located, so nothing here resolves or claims a scheme. Defaults to
            :class:`~molhub.sources.drivers.https.HttpsSource`, so cached
            downloads inherit the same status-checked, atomically-renamed
            transfer as everything else.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        transfer: FileTransfer | None = None,
    ) -> None:
        self._root = BlobStore(root).root
        self._transfer = transfer or HttpsSource()

    @property
    def root(self) -> Path:
        """The cache root — the same directory :class:`BlobStore` uses."""
        return self._root

    def path_for(self, url: str) -> Path:
        """Return a URL-specific cache path without touching the network.

        The readable filename tail is bounded to leave space for the URL ID
        and the transfer's temporary-file suffix on common filesystems.
        """
        url_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
        filename = self.filename_for(url)[-128:]
        return self._root / _URL_SUBDIR / f"{url_id}-{filename}"

    def fetch(self, url: str) -> Path:
        """Return a local path for *url*, downloading it if not already cached.

        Raises:
            BadStatus: If the server responds with a status other than 200.
        """
        destination = self.path_for(url)
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._transfer.fetch(RemoteFile(url=url, filename=destination.name), destination)
        return destination

    @staticmethod
    def filename_for(url: str) -> str:
        """Best-effort filename from a URL, ignoring any query string.

        The result is always a single path segment: anything that could escape
        ``urls/`` or upset a filesystem is replaced, and a URL that yields no
        usable name falls back to ``"download"``.
        """
        name = _UNSAFE.sub("_", url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1])
        return name if name.strip(".") else "download"
