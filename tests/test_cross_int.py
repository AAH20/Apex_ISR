"""Tests for cross-INT correlation, entity disambiguation, and identity resolution.

Written test-first (TDD). Covers:
- Cross-INT correlation by name, location, time, and attributes
- Entity disambiguation among candidates
- Identity resolution across INT sources
"""

from __future__ import annotations

import pytest

from src.isr.cross_int import (
    CorrelationType,
    CrossIntCorrelator,
    DisambiguationResult,
    EntityDisambiguator,
    IdentityResolution,
    IdentityResolver,
)
from src.isr.entity_resolution import (
    EntityObservation,
    EntityResolver,
    EntityType,
    INTSource,
    ResolvedEntity,
    normalize_name,
)


def make_obs(
    name,
    entity_type=EntityType.PERSON,
    source=INTSource.OSINT,
    aliases=(),
    attributes=None,
    location=None,
    reliability=None,
    timestamp=None,
):
    obs = EntityObservation(
        source=source,
        entity_type=entity_type,
        name=name,
        aliases=tuple(aliases),
        attributes=dict(attributes or {}),
        location=location,
        reliability=reliability,
    )
    if timestamp is not None:
        obs.attributes["timestamp"] = timestamp
    return obs


def make_entity(
    name,
    entity_type=EntityType.PERSON,
    aliases=(),
    attributes=None,
    location=None,
    source=INTSource.OSINT,
    reliability=None,
):
    obs = make_obs(name, entity_type, source, aliases, attributes, location, reliability)
    return ResolvedEntity(
        entity_id=f"e-{normalize_name(name).replace(' ', '-')}",
        entity_type=entity_type,
        names={name},
        aliases=set(aliases),
        attributes={k: {str(v)} for k, v in (attributes or {}).items()},
        observations=[obs],
    )


# ---------------------------------------------------------------------------
# Cross-INT Correlation tests
# ---------------------------------------------------------------------------


class TestCrossIntCorrelationByName:
    def test_correlates_entities_with_same_name(self):
        correlator = CrossIntCorrelator()
        alice_osint = make_entity("Alice Smith", source=INTSource.OSINT)
        alice_geoint = make_entity("Alice Smith", source=INTSource.GEOINT)
        correlations = correlator.correlate_by_name([alice_osint, alice_geoint])
        assert len(correlations) >= 1
        assert correlations[0].correlation_type == CorrelationType.IDENTITY

    def test_no_correlation_for_different_names(self):
        correlator = CrossIntCorrelator()
        alice = make_entity("Alice Smith", source=INTSource.OSINT)
        bob = make_entity("Bob Jones", source=INTSource.GEOINT)
        correlations = correlator.correlate_by_name([alice, bob])
        assert len(correlations) == 0

    def test_correlation_includes_both_sources(self):
        correlator = CrossIntCorrelator()
        alice_osint = make_entity("Alice Smith", source=INTSource.OSINT)
        alice_geoint = make_entity("Alice Smith", source=INTSource.GEOINT)
        correlations = correlator.correlate_by_name([alice_osint, alice_geoint])
        assert len(correlations) == 1
        assert correlations[0].sources == {INTSource.OSINT, INTSource.GEOINT}

    def test_correlation_confidence_is_bounded(self):
        correlator = CrossIntCorrelator()
        alice_osint = make_entity("Alice Smith", source=INTSource.OSINT)
        alice_geoint = make_entity("Alice Smith", source=INTSource.GEOINT)
        correlations = correlator.correlate_by_name([alice_osint, alice_geoint])
        for c in correlations:
            assert 0.0 <= c.confidence <= 1.0


class TestCrossIntCorrelationByLocation:
    def test_correlates_entities_in_proximity(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity("Entity A", location=(40.0, -73.0), source=INTSource.OSINT)
        entity_b = make_entity("Entity B", location=(40.01, -73.01), source=INTSource.GEOINT)
        correlations = correlator.correlate_by_location([entity_a, entity_b], max_distance_km=50.0)
        assert len(correlations) >= 1
        assert correlations[0].correlation_type == CorrelationType.SPATIAL

    def test_no_correlation_for_distant_entities(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity("Entity A", location=(40.0, -73.0), source=INTSource.OSINT)
        entity_b = make_entity("Entity B", location=(51.5, -0.12), source=INTSource.GEOINT)
        correlations = correlator.correlate_by_location([entity_a, entity_b], max_distance_km=50.0)
        assert len(correlations) == 0

    def test_location_correlation_includes_spatial_type(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity("Alpha", location=(40.0, -73.0), source=INTSource.OSINT)
        entity_b = make_entity("Beta", location=(40.005, -73.005), source=INTSource.GEOINT)
        correlations = correlator.correlate_by_location([entity_a, entity_b])
        assert all(c.correlation_type == CorrelationType.SPATIAL for c in correlations)


class TestCrossIntCorrelationByTime:
    def test_correlates_entities_with_close_timestamps(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity("Entity A", source=INTSource.OSINT, attributes={"timestamp": 1000.0})
        entity_b = make_entity("Entity B", source=INTSource.GEOINT, attributes={"timestamp": 1100.0})
        correlations = correlator.correlate_by_time([entity_a, entity_b], time_window_hours=1.0)
        assert len(correlations) >= 1
        assert correlations[0].correlation_type == CorrelationType.TEMPORAL

    def test_no_correlation_for_distant_timestamps(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity("Entity A", source=INTSource.OSINT, attributes={"timestamp": 1000.0})
        entity_b = make_entity("Entity B", source=INTSource.GEOINT, attributes={"timestamp": 5000.0})
        correlations = correlator.correlate_by_time([entity_a, entity_b], time_window_hours=1.0)
        assert len(correlations) == 0


class TestCrossIntCorrelationByAttribute:
    def test_correlates_entities_sharing_attributes(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity("Entity A", source=INTSource.OSINT, attributes={"org": "Alpha Group"})
        entity_b = make_entity("Entity B", source=INTSource.GEOINT, attributes={"org": "Alpha Group"})
        correlations = correlator.correlate_by_attribute([entity_a, entity_b])
        assert len(correlations) >= 1
        assert correlations[0].correlation_type == CorrelationType.ORGANIZATIONAL

    def test_no_correlation_for_disjoint_attributes(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity("Entity A", source=INTSource.OSINT, attributes={"org": "Alpha Group"})
        entity_b = make_entity("Entity B", source=INTSource.GEOINT, attributes={"org": "Beta Group"})
        correlations = correlator.correlate_by_attribute([entity_a, entity_b])
        assert len(correlations) == 0


class TestFindAllCorrelations:
    def test_find_all_correlations_aggregates_types(self):
        correlator = CrossIntCorrelator()
        entity_a = make_entity(
            "Alice Smith",
            source=INTSource.OSINT,
            location=(40.0, -73.0),
            attributes={"timestamp": 1000.0, "rank": "captain"},
        )
        entity_b = make_entity(
            "Alice Smith",
            source=INTSource.GEOINT,
            location=(40.01, -73.01),
            attributes={"timestamp": 1100.0, "rank": "captain"},
        )
        correlations = correlator.find_all_correlations([entity_a, entity_b])
        assert len(correlations) >= 1
        types = {c.correlation_type for c in correlations}
        assert CorrelationType.IDENTITY in types

    def test_find_all_correlations_empty_for_single_entity(self):
        correlator = CrossIntCorrelator()
        entity = make_entity("Alice Smith", source=INTSource.OSINT)
        correlations = correlator.find_all_correlations([entity])
        assert len(correlations) == 0


# ---------------------------------------------------------------------------
# Entity Disambiguation tests
# ---------------------------------------------------------------------------


class TestEntityDisambiguation:
    def test_disambiguate_clear_winner(self):
        disambiguator = EntityDisambiguator()
        alice = make_entity("Alice Smith", attributes={"rank": "captain"})
        bob = make_entity("Bob Jones")
        obs = make_obs("Alice Smith", attributes={"rank": "captain"}, source=INTSource.GEOINT)
        result = disambiguator.disambiguate(obs, [bob, alice])
        assert result.best_match is alice
        assert not result.is_ambiguous

    def test_disambiguate_ambiguous_when_scores_close(self):
        disambiguator = EntityDisambiguator()
        alice = make_entity("Alice Smith")
        alicia = make_entity("Alicia Smith")
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        result = disambiguator.disambiguate(obs, [alice, alicia])
        assert result.is_ambiguous

    def test_disambiguate_no_candidates(self):
        disambiguator = EntityDisambiguator()
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        result = disambiguator.disambiguate(obs, [])
        assert result.best_match is None
        assert result.candidates == []

    def test_disambiguate_returns_scores_for_all_candidates(self):
        disambiguator = EntityDisambiguator()
        alice = make_entity("Alice Smith")
        bob = make_entity("Bob Jones")
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        result = disambiguator.disambiguate(obs, [alice, bob])
        assert len(result.scores) == 2
        assert alice.entity_id in result.scores
        assert bob.entity_id in result.scores

    def test_is_ambiguous_true_when_top_two_within_threshold(self):
        disambiguator = EntityDisambiguator()
        scores = {"e-a": 0.85, "e-b": 0.80}
        assert disambiguator.is_ambiguous(scores, threshold=0.1)

    def test_is_ambiguous_false_when_clear_gap(self):
        disambiguator = EntityDisambiguator()
        scores = {"e-a": 0.90, "e-b": 0.30}
        assert not disambiguator.is_ambiguous(scores, threshold=0.1)

    def test_is_ambiguous_false_for_single_candidate(self):
        disambiguator = EntityDisambiguator()
        scores = {"e-a": 0.90}
        assert not disambiguator.is_ambiguous(scores, threshold=0.1)


# ---------------------------------------------------------------------------
# Identity Resolution tests
# ---------------------------------------------------------------------------


class TestIdentityResolution:
    def test_resolve_identity_merges_observation(self):
        resolver = IdentityResolver()
        entity = make_entity("Alice Smith", source=INTSource.OSINT)
        obs = make_obs("Alice Smith", source=INTSource.GEOINT, attributes={"rank": "captain"})
        result = resolver.resolve_identity(obs, [entity])
        assert result.canonical_entity is entity
        assert obs in result.merged_observations
        assert result.source_coverage == {INTSource.OSINT, INTSource.GEOINT}

    def test_resolve_identity_creates_new_when_no_match(self):
        resolver = IdentityResolver()
        obs = make_obs("Alice Smith", source=INTSource.OSINT)
        result = resolver.resolve_identity(obs, [])
        assert result.canonical_entity is not None
        assert obs in result.merged_observations
        assert result.source_coverage == {INTSource.OSINT}

    def test_resolve_identity_confidence_increases_with_sources(self):
        resolver = IdentityResolver()
        entity = make_entity("Alice Smith", source=INTSource.OSINT)
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        result = resolver.resolve_identity(obs, [entity])
        assert result.confidence > 0.0
        assert result.confidence <= 1.0

    def test_resolve_identity_collects_aliases(self):
        resolver = IdentityResolver()
        entity = make_entity("Alice Smith", aliases=("A. Smith",), source=INTSource.OSINT)
        obs = make_obs("Alice Smith", aliases=("Alice S.",), source=INTSource.GEOINT)
        result = resolver.resolve_identity(obs, [entity])
        assert "A. Smith" in result.aliases
        assert "Alice S." in result.aliases

    def test_merge_identities_combines_observations(self):
        resolver = IdentityResolver()
        entity_a = make_entity("Alice Smith", source=INTSource.OSINT)
        entity_b = make_entity("Alice Smith", source=INTSource.GEOINT)
        result = resolver.merge_identities([entity_a, entity_b])
        assert len(result.merged_observations) == 2
        assert result.source_coverage == {INTSource.OSINT, INTSource.GEOINT}

    def test_merge_identities_unifies_names(self):
        resolver = IdentityResolver()
        entity_a = make_entity("Alice Smith", source=INTSource.OSINT)
        entity_b = make_entity("A. Smith", source=INTSource.GEOINT)
        result = resolver.merge_identities([entity_a, entity_b])
        assert "Alice Smith" in result.canonical_entity.names
        assert "A. Smith" in result.canonical_entity.names

    def test_resolve_all_groups_observations_by_identity(self):
        resolver = IdentityResolver()
        obs1 = make_obs("Alice Smith", source=INTSource.OSINT)
        obs2 = make_obs("Alice Smith", source=INTSource.GEOINT)
        obs3 = make_obs("Bob Jones", source=INTSource.OSINT)
        results = resolver.resolve_all([obs1, obs2, obs3])
        assert len(results) == 2
        # Find the Alice group
        alice_results = [r for r in results if any("Alice" in n for n in r.canonical_entity.names)]
        assert len(alice_results) == 1
        assert len(alice_results[0].merged_observations) == 2

    def test_resolve_all_empty_input(self):
        resolver = IdentityResolver()
        results = resolver.resolve_all([])
        assert results == []

    def test_identity_resolution_with_different_types_creates_separate(self):
        resolver = IdentityResolver()
        person = make_entity("Alice Smith", entity_type=EntityType.PERSON, source=INTSource.OSINT)
        vessel_obs = make_obs("Alice Smith", entity_type=EntityType.VESSEL, source=INTSource.GEOINT)
        result = resolver.resolve_identity(vessel_obs, [person])
        assert result.canonical_entity is not person
        assert result.canonical_entity.entity_type == EntityType.VESSEL
