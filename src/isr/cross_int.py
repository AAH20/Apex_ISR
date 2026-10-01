"""Cross-INT correlation, entity disambiguation, and identity resolution.

Deepens multi-INT fusion by correlating entities across intelligence
disciplines (HUMINT/SIGINT/GEOINT/OSINT/MASINT/CYBINT), disambiguating
among candidate matches, and resolving identities with confidence
scoring and alias collection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Optional

from src.isr.entity_resolution import (
    EntityObservation,
    EntityResolver,
    EntityType,
    INTSource,
    ResolvedEntity,
    normalize_name,
)


class CorrelationType(Enum):
    """Types of cross-INT correlations."""

    IDENTITY = "identity"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    ORGANIZATIONAL = "organizational"


@dataclass
class Correlation:
    """A correlation between two resolved entities across INT sources."""

    entity_a: ResolvedEntity
    entity_b: ResolvedEntity
    correlation_type: CorrelationType
    confidence: float
    sources: set[INTSource] = field(default_factory=set)


@dataclass
class DisambiguationResult:
    """Result of disambiguating an observation among candidate entities."""

    best_match: Optional[ResolvedEntity]
    is_ambiguous: bool
    candidates: list[ResolvedEntity] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class IdentityResolution:
    """Result of identity resolution across INT sources."""

    canonical_entity: ResolvedEntity
    merged_observations: list[EntityObservation] = field(default_factory=list)
    source_coverage: set[INTSource] = field(default_factory=set)
    confidence: float = 0.0
    aliases: set[str] = field(default_factory=set)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


class CrossIntCorrelator:
    """Correlates resolved entities across INT sources by multiple dimensions."""

    DEFAULT_MAX_DISTANCE_KM = 100.0
    DEFAULT_TIME_WINDOW_HOURS = 24.0
    NAME_SIMILARITY_THRESHOLD = 0.85

    def __init__(self, resolver: Optional[EntityResolver] = None):
        self.resolver = resolver or EntityResolver()

    def correlate_by_name(
        self, entities: list[ResolvedEntity]
    ) -> list[Correlation]:
        """Find entity pairs with similar names across different INT sources."""
        correlations: list[Correlation] = []
        for i, entity_a in enumerate(entities):
            for entity_b in entities[i + 1 :]:
                if entity_a is entity_b:
                    continue
                if not self._shares_source(entity_a, entity_b):
                    continue
                sim = self._name_similarity(entity_a, entity_b)
                if sim >= self.NAME_SIMILARITY_THRESHOLD:
                    sources = self._entity_sources(entity_a) | self._entity_sources(entity_b)
                    correlations.append(
                        Correlation(
                            entity_a=entity_a,
                            entity_b=entity_b,
                            correlation_type=CorrelationType.IDENTITY,
                            confidence=sim,
                            sources=sources,
                        )
                    )
        return correlations

    def correlate_by_location(
        self,
        entities: list[ResolvedEntity],
        max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    ) -> list[Correlation]:
        """Find entity pairs within spatial proximity."""
        correlations: list[Correlation] = []
        for i, entity_a in enumerate(entities):
            for entity_b in entities[i + 1 :]:
                if entity_a is entity_b:
                    continue
                if not self._shares_source(entity_a, entity_b):
                    continue
                dist = self._min_distance_km(entity_a, entity_b)
                if dist is not None and dist <= max_distance_km:
                    confidence = math.exp(-dist / max_distance_km)
                    sources = self._entity_sources(entity_a) | self._entity_sources(entity_b)
                    correlations.append(
                        Correlation(
                            entity_a=entity_a,
                            entity_b=entity_b,
                            correlation_type=CorrelationType.SPATIAL,
                            confidence=confidence,
                            sources=sources,
                        )
                    )
        return correlations

    def correlate_by_time(
        self,
        entities: list[ResolvedEntity],
        time_window_hours: float = DEFAULT_TIME_WINDOW_HOURS,
    ) -> list[Correlation]:
        """Find entity pairs with observations close in time."""
        correlations: list[Correlation] = []
        window_seconds = time_window_hours * 3600.0
        for i, entity_a in enumerate(entities):
            for entity_b in entities[i + 1 :]:
                if entity_a is entity_b:
                    continue
                if not self._shares_source(entity_a, entity_b):
                    continue
                time_diff = self._min_time_diff(entity_a, entity_b)
                if time_diff is not None and time_diff <= window_seconds:
                    confidence = 1.0 - (time_diff / window_seconds)
                    sources = self._entity_sources(entity_a) | self._entity_sources(entity_b)
                    correlations.append(
                        Correlation(
                            entity_a=entity_a,
                            entity_b=entity_b,
                            correlation_type=CorrelationType.TEMPORAL,
                            confidence=confidence,
                            sources=sources,
                        )
                    )
        return correlations

    def correlate_by_attribute(
        self, entities: list[ResolvedEntity]
    ) -> list[Correlation]:
        """Find entity pairs sharing attribute key/value pairs."""
        correlations: list[Correlation] = []
        for i, entity_a in enumerate(entities):
            for entity_b in entities[i + 1 :]:
                if entity_a is entity_b:
                    continue
                if not self._shares_source(entity_a, entity_b):
                    continue
                shared = self._shared_attributes(entity_a, entity_b)
                if shared:
                    confidence = len(shared) / max(
                        len(entity_a.attributes), len(entity_b.attributes), 1
                    )
                    sources = self._entity_sources(entity_a) | self._entity_sources(entity_b)
                    correlations.append(
                        Correlation(
                            entity_a=entity_a,
                            entity_b=entity_b,
                            correlation_type=CorrelationType.ORGANIZATIONAL,
                            confidence=confidence,
                            sources=sources,
                        )
                    )
        return correlations

    def find_all_correlations(
        self, entities: list[ResolvedEntity]
    ) -> list[Correlation]:
        """Find all correlations across all dimensions."""
        all_correlations: list[Correlation] = []
        all_correlations.extend(self.correlate_by_name(entities))
        all_correlations.extend(self.correlate_by_location(entities))
        all_correlations.extend(self.correlate_by_time(entities))
        all_correlations.extend(self.correlate_by_attribute(entities))
        return all_correlations

    def _shares_source(
        self, entity_a: ResolvedEntity, entity_b: ResolvedEntity
    ) -> bool:
        """True if the two entities come from different INT sources."""
        sources_a = self._entity_sources(entity_a)
        sources_b = self._entity_sources(entity_b)
        return sources_a != sources_b

    def _entity_sources(self, entity: ResolvedEntity) -> set[INTSource]:
        """Get the set of INT sources for an entity."""
        return {obs.source for obs in entity.observations}

    def _name_similarity(
        self, entity_a: ResolvedEntity, entity_b: ResolvedEntity
    ) -> float:
        """Maximum name similarity between two entities."""
        names_a = {normalize_name(n) for n in entity_a.names}
        names_a |= {normalize_name(a) for a in entity_a.aliases}
        names_b = {normalize_name(n) for n in entity_b.names}
        names_b |= {normalize_name(a) for a in entity_b.aliases}
        if not names_a or not names_b:
            return 0.0
        best = 0.0
        for na in names_a:
            for nb in names_b:
                if na == nb:
                    return 1.0
                sim = SequenceMatcher(None, na, nb).ratio()
                best = max(best, sim)
        return best

    def _min_distance_km(
        self, entity_a: ResolvedEntity, entity_b: ResolvedEntity
    ) -> Optional[float]:
        """Minimum distance between any two observation locations."""
        locs_a = [o.location for o in entity_a.observations if o.location is not None]
        locs_b = [o.location for o in entity_b.observations if o.location is not None]
        if not locs_a or not locs_b:
            return None
        return min(_haversine_km(a, b) for a in locs_a for b in locs_b)

    def _min_time_diff(
        self, entity_a: ResolvedEntity, entity_b: ResolvedEntity
    ) -> Optional[float]:
        """Minimum absolute time difference between observations."""
        times_a = self._entity_timestamps(entity_a)
        times_b = self._entity_timestamps(entity_b)
        if not times_a or not times_b:
            return None
        return min(abs(a - b) for a in times_a for b in times_b)

    def _entity_timestamps(self, entity: ResolvedEntity) -> list[float]:
        """Extract timestamps from entity observations."""
        timestamps = []
        for obs in entity.observations:
            ts = obs.attributes.get("timestamp")
            if ts is not None:
                try:
                    timestamps.append(float(ts))
                except (ValueError, TypeError):
                    pass
        return timestamps

    def _shared_attributes(
        self, entity_a: ResolvedEntity, entity_b: ResolvedEntity
    ) -> set[str]:
        """Find attribute keys shared by both entities with matching values."""
        shared = set()
        for key, values_a in entity_a.attributes.items():
            if key in entity_b.attributes:
                values_b = entity_b.attributes[key]
                if values_a & values_b:
                    shared.add(key)
        return shared


class EntityDisambiguator:
    """Disambiguates observations among candidate resolved entities."""

    AMBIGUITY_THRESHOLD = 0.1

    def __init__(self, resolver: Optional[EntityResolver] = None):
        self.resolver = resolver or EntityResolver()

    def disambiguate(
        self, obs: EntityObservation, entities: list[ResolvedEntity]
    ) -> DisambiguationResult:
        """Disambiguate an observation among candidate entities."""
        if not entities:
            return DisambiguationResult(
                best_match=None,
                is_ambiguous=False,
                candidates=[],
                scores={},
            )
        scores: dict[str, float] = {}
        for entity in entities:
            scores[entity.entity_id] = self.resolver.score_observation(obs, entity)
        best = max(entities, key=lambda e: scores[e.entity_id])
        ambiguous = self.is_ambiguous(scores, self.AMBIGUITY_THRESHOLD)
        return DisambiguationResult(
            best_match=best,
            is_ambiguous=ambiguous,
            candidates=list(entities),
            scores=scores,
        )

    def is_ambiguous(
        self, scores: dict[str, float], threshold: float = AMBIGUITY_THRESHOLD
    ) -> bool:
        """True if the top two scores are within the ambiguity threshold."""
        if len(scores) < 2:
            return False
        sorted_scores = sorted(scores.values(), reverse=True)
        return (sorted_scores[0] - sorted_scores[1]) <= threshold


class IdentityResolver:
    """Resolves identities across INT sources with confidence scoring."""

    def __init__(self, resolver: Optional[EntityResolver] = None):
        self.resolver = resolver or EntityResolver()

    def resolve_identity(
        self, obs: EntityObservation, entities: list[ResolvedEntity]
    ) -> IdentityResolution:
        """Resolve an observation to a canonical entity, merging if matched."""
        match = self.resolver.match_observation(obs, entities)
        if match is not None:
            self.resolver._merge_observation(match.entity, obs)
            return self._build_resolution(match.entity, [obs])
        # No match — create new entity
        new_entity = self.resolver._create_entity(obs)
        return self._build_resolution(new_entity, [obs])

    def merge_identities(
        self, entities: list[ResolvedEntity]
    ) -> IdentityResolution:
        """Merge multiple resolved entities into one canonical identity."""
        if not entities:
            raise ValueError("Cannot merge empty entity list")
        canonical = entities[0]
        all_obs: list[EntityObservation] = []
        all_aliases: set[str] = set()
        for entity in entities:
            for obs in entity.observations:
                if obs not in all_obs:
                    all_obs.append(obs)
            all_aliases |= entity.aliases
            all_aliases |= entity.names
        # Merge all into canonical
        for entity in entities[1:]:
            for obs in entity.observations:
                if obs not in canonical.observations:
                    canonical.observations.append(obs)
            canonical.names |= entity.names
            canonical.aliases |= entity.aliases
            for key, values in entity.attributes.items():
                canonical.attributes.setdefault(key, set()).update(values)
        canonical.aliases |= all_aliases
        return self._build_resolution(canonical, all_obs)

    def resolve_all(
        self, observations: list[EntityObservation]
    ) -> list[IdentityResolution]:
        """Resolve all observations, grouping by identity."""
        if not observations:
            return []
        entities: list[ResolvedEntity] = []
        for obs in observations:
            match = self.resolver.match_observation(obs, entities)
            if match is not None:
                self.resolver._merge_observation(match.entity, obs)
            else:
                entities.append(self.resolver._create_entity(obs))
        # Build one IdentityResolution per canonical entity
        results: list[IdentityResolution] = []
        for entity in entities:
            results.append(self._build_resolution(entity, list(entity.observations)))
        return results

    def _build_resolution(
        self, entity: ResolvedEntity, observations: list[EntityObservation]
    ) -> IdentityResolution:
        """Build an IdentityResolution from a canonical entity."""
        sources = {obs.source for obs in entity.observations}
        confidence = self.resolver.confidence(entity)
        aliases = set(entity.aliases)
        for obs in observations:
            aliases |= set(obs.aliases)
        return IdentityResolution(
            canonical_entity=entity,
            merged_observations=list(observations),
            source_coverage=sources,
            confidence=confidence,
            aliases=aliases,
        )
