"""Windows filesystem paths without legacy length limits or display prefixes.

Normalize only at the filesystem boundary. Do not resolve symlinks here: callers
that enforce provenance or containment must still be able to inspect them.
"""
from __future__ import annotations

import ntpath
import os
from pathlib import Path


WINDOWS = os.name == 'nt'


def filesystem_path(value: str | os.PathLike[str]) -> Path:
    """Return a usable absolute extended Windows path, or an ordinary POSIX Path.

    The explicit namespace works without requiring a system registry change.
    Relative/drive-relative paths use the current Windows working directory;
    existing extended paths and UNC shares retain their filesystem identity.
    """
    if not WINDOWS:
        return Path(value)
    text = os.fspath(value)
    if text.startswith('\\\\?\\'):
        return Path(text)
    absolute = ntpath.normpath(ntpath.abspath(text))
    # Windows can return an extended path for a relative input when the current
    # working directory is already extended. It is not an ordinary UNC share.
    if absolute.startswith('\\\\?\\'):
        return Path(absolute)
    if absolute.startswith('\\\\'):
        return Path('\\\\?\\UNC\\' + absolute[2:])
    return Path('\\\\?\\' + absolute)


def display_path(value: str | os.PathLike[str]) -> str:
    """Keep the internal Windows namespace out of filenames shown to operators."""
    text = os.fspath(value)
    if text[:8].lower() == '\\\\?\\unc\\':
        return '\\\\' + text[8:]
    if text.startswith('\\\\?\\'):
        return text[4:]
    return text
