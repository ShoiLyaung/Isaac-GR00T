"""Framework-neutral validation for causal episode-bundle views."""

from __future__ import annotations

from .episode_bundle import EpisodeT8View, validate_episode_view


class CodecEvidenceProcessorCore:
    """Validate and preserve one exact causal T8 episode view."""

    def process(self, view: EpisodeT8View) -> EpisodeT8View:
        if not isinstance(view, EpisodeT8View):
            raise TypeError("Only codec-evidence-episode-bundle-v1 views are supported")
        validate_episode_view(view)
        return view


__all__ = ["CodecEvidenceProcessorCore"]
