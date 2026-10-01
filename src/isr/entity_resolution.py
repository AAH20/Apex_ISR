"""Multi-INT entity resolution.

Cross-INT entity matching, disambiguation, identity resolution, and
confidence scoring across HUMINT/SIGINT/GEOINT/OSINT/MASINT/CYBINT sources.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Optional


class EntityType(Enum):
    """Types of entities resolvable across INT sources."""

    PERSON = "person"
    VESSEL = "vessel"
    AIRCRAFT = "aircraft"
    VEHICLE = "vehicle"
    ORGANIZATION = "organization"
    FACILITY = "facility"
    LOCATION = "location"
    UNKNOWN = "unknown"


class INTSource(Enum):
    """Intelligence disciplines that produce observations."""

    HUMINT = "humint"
    SIGINT = "sigint"
    GEOINT = "geoint"
    OSINT = "osint"
    MASINT = "masint"
    CYBINT = "cybint"


def normalize_name(name: str) -> str:
    """Normalize a name for comparison: lowercase, strip punctuation, collapse whitespace."""
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    return " ".join(cleaned.split())


@dataclass
class EntityObservation:
    """A single observation of an entity from one INT source."""

    source: INTSource
    entity_type: EntityType
    name: str
    aliases: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    location: Optional[tuple[float, float]] = None
    reliability: Optional[float] = None


@dataclass
class ResolvedEntity:
    """A resolved entity aggregating observations across INT sources."""

    entity_id: str
    entity_type: EntityType
    names: set[str] = field(default_factory=set)
    aliases: set[str] = field(default_factory=set)
    attributes: dict[str, set[str]] = field(default_factory=dict)
    observations: list[EntityObservation] = field(default_factory=list)


@dataclass
class MatchResult:
    """A candidate match between an observation and a resolved entity."""

    entity: ResolvedEntity
    score: float


class EntityResolver:
    """Resolves entities across INT sources with confidence scoring."""

    NAME_WEIGHT = 0.45
    TYPE_WEIGHT = 0.20
    ATTRIBUTE_WEIGHT = 0.20
    LOCATION_WEIGHT = 0.15
    DEFAULT_RELIABILITY = 0.5
    LOCATION_SCALE_KM = 100.0
    MATCH_THRESHOLD = 0.4
    LOCATION_CONFLICT_PENALTY = 0.5
    LOCATION_CONFLICT_KM = 500.0

    def name_similarity(self, obs: EntityObservation, entity: ResolvedEntity) -> float:
        """Similarity between an observation name and a resolved entity (0..1)."""
        obs_norm = normalize_name(obs.name)
        if not obs_norm:
            return 0.0
        candidates = {normalize_name(n) for n in entity.names}
        candidates |= {normalize_name(a) for a in entity.aliases}
        if obs_norm in candidates:
            return 1.0
        if not candidates:
            return 0.0
        return max(SequenceMatcher(None, obs_norm, c).ratio() for c in candidates)

    def type_similarity(self, obs: EntityObservation, entity: ResolvedEntity) -> float:
        """Compatibility between observation and entity types (0..1)."""
        if obs.entity_type == entity.entity_type:
            return 1.0
        if EntityType.UNKNOWN in (obs.entity_type, entity.entity_type):
            return 0.5
        return 0.0

    def _attribute_similarity(self, obs: EntityObservation, entity: ResolvedEntity) -> float:
        """Jaccard-style overlap of attribute key/value pairs (0..1)."""
        if not obs.attributes or not entity.attributes:
            return 0.0
        obs_pairs = {k: str(v) for k, v in obs.attributes.items()}
        matches = sum(
            1 for k, v in obs_pairs.items() if k in entity.attributes and v in entity.attributes[k]
        )
        union = len(obs_pairs) + sum(
            1 for k in entity.attributes if k not in obs_pairs
        )
        return matches / union if union else 0.0

    def _location_similarity(self, obs: EntityObservation, entity: ResolvedEntity) -> float:
        """Proximity of observation location to entity locations (0..1)."""
        if obs.location is None:
            return 0.0
        entity_locs = [o.location for o in entity.observations if o.location is not None]
        if not entity_locs:
            return 0.0
        best = max(self._haversine_km(obs.location, loc) for loc in entity_locs)
        return math.exp(-best / self.LOCATION_SCALE_KM)

    def _location_conflict(
        self, obs: EntityObservation, entity: ResolvedEntity
    ) -> bool:
        """True if the observation location conflicts with all entity locations."""
        if obs.location is None:
            return False
        entity_locs = [o.location for o in entity.observations if o.location is not None]
        if not entity_locs:
            return False
        return all(
            self._haversine_km(obs.location, loc) > self.LOCATION_CONFLICT_KM
            for loc in entity_locs
        )

    @staticmethod
    def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
        """Great-circle distance in km between two (lat, lon) points."""
        lat1, lon1 = math.radians(a[0]), math.radians(a[1])
        lat2, lon2 = math.radians(b[0]), math.radians(b[1])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371.0 * 2 * math.asin(math.sqrt(h))

    def score_observation(self, obs: EntityObservation, entity: ResolvedEntity) -> float:
        """Weighted match score of an observation against an entity (0..1)."""
        type_sim = self.type_similarity(obs, entity)
        if type_sim == 0.0:
            return 0.0
        base = (
            self.NAME_WEIGHT * self.name_similarity(obs, entity)
            + self.TYPE_WEIGHT * type_sim
            + self.ATTRIBUTE_WEIGHT * self._attribute_similarity(obs, entity)
            + self.LOCATION_WEIGHT * self._location_similarity(obs, entity)
        )
        reliability = (
            obs.reliability
            if obs.reliability is not None
            else self.DEFAULT_RELIABILITY
        )
        score = base * (0.5 + 0.5 * reliability)
        if self._location_conflict(obs, entity):
            score *= self.LOCATION_CONFLICT_PENALTY
        return score

    def match_observation(
        self, obs: EntityObservation, entities: list[ResolvedEntity]
    ) -> MatchResult | None:
        """Return the best-scoring entity for an observation, or None if below threshold."""
        if not entities:
            return None
        best = max(entities, key=lambda e: self.score_observation(obs, e))
        score = self.score_observation(obs, best)
        if score < self.MATCH_THRESHOLD:
            return None
        return MatchResult(entity=best, score=score)

    def disambiguate(
        self, obs: EntityObservation, entities: list[ResolvedEntity]
    ) -> list[MatchResult]:
        """Rank all candidate entities by match score, best first."""
        results = [
            MatchResult(entity=e, score=self.score_observation(obs, e)) for e in entities
        ]
        return sorted(results, key=lambda r: r.score, reverse=True)

    def resolve(
        self, obs: EntityObservation, entities: list[ResolvedEntity]
    ) -> ResolvedEntity:
        """Resolve an observation: merge into best match or create a new entity."""
        match = self.match_observation(obs, entities)
        if match is not None:
            self._merge_observation(match.entity, obs)
            return match.entity
        return self._create_entity(obs)

    def confidence(self, entity: ResolvedEntity) -> float:
        """Confidence in a resolved entity based on observation count and source diversity (0..1)."""
        n = len(entity.observations)
        if n == 0:
            return 0.0
        sources = {o.source for o in entity.observations}
        source_diversity = len(sources) / len(INTSource)
        count_factor = 1.0 - 1.0 / (1.0 + n)
        return min(1.0, count_factor * (0.5 + 0.5 * source_diversity))

    def _merge_observation(
        self, entity: ResolvedEntity, obs: EntityObservation
    ) -> None:
        """Merge an observation into an existing entity."""
        if obs in entity.observations:
            return
        entity.observations.append(obs)
        entity.names.add(obs.name)
        entity.aliases.update(obs.aliases)
        for key, value in obs.attributes.items():
            entity.attributes.setdefault(key, set()).add(str(value))

    def _create_entity(self, obs: EntityObservation) -> ResolvedEntity:
        """Create a new resolved entity from a single observation."""
        entity = ResolvedEntity(
            entity_id=f"e-{normalize_name(obs.name).replace(' ', '-')}",
            entity_type=obs.entity_type,
            names={obs.name},
            aliases=set(obs.aliases),
            attributes={k: {str(v)} for k, v in obs.attributes.items()},
            observations=[obs],
        )
        return entity
