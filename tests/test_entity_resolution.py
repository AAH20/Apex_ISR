"""Tests for multi-INT entity resolution.

Covers cross-INT entity matching, disambiguation, identity resolution,
and confidence scoring. Written test-first (TDD).
"""

from __future__ import annotations

import pytest

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
):
    return EntityObservation(
        source=source,
        entity_type=entity_type,
        name=name,
        aliases=tuple(aliases),
        attributes=dict(attributes or {}),
        location=location,
        reliability=reliability,
    )


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


class TestNormalizeName:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize_name("  O'Brien,  JOHN ") == "o brien john"

    def test_collapses_internal_whitespace(self):
        assert normalize_name("Alice   Smith\tJones\n") == "alice smith jones"


class TestNameSimilarity:
    def test_exact_match_after_normalization_scores_one(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        obs = make_obs("  alice   SMITH ")
        assert resolver.name_similarity(obs, entity) == pytest.approx(1.0)

    def test_alias_match_scores_one(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith", aliases=("A. Smith",))
        obs = make_obs("a smith")
        assert resolver.name_similarity(obs, entity) == pytest.approx(1.0)

    def test_different_names_score_below_one(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        obs = make_obs("Bob Jones")
        assert resolver.name_similarity(obs, entity) < 1.0

    def test_empty_names_score_zero(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        obs = make_obs("")
        assert resolver.name_similarity(obs, entity) == 0.0


class TestTypeSimilarity:
    def test_same_type_scores_one(self):
        resolver = EntityResolver()
        entity = make_entity("X", entity_type=EntityType.VESSEL)
        obs = make_obs("Y", entity_type=EntityType.VESSEL)
        assert resolver.type_similarity(obs, entity) == pytest.approx(1.0)

    def test_unknown_type_is_partially_compatible(self):
        resolver = EntityResolver()
        entity = make_entity("X", entity_type=EntityType.PERSON)
        obs = make_obs("Y", entity_type=EntityType.UNKNOWN)
        assert resolver.type_similarity(obs, entity) == pytest.approx(0.5)

    def test_different_types_score_zero(self):
        resolver = EntityResolver()
        entity = make_entity("X", entity_type=EntityType.PERSON)
        obs = make_obs("Y", entity_type=EntityType.VESSEL)
        assert resolver.type_similarity(obs, entity) == 0.0


class TestScoreObservation:
    def test_type_mismatch_penalizes_score(self):
        resolver = EntityResolver()
        same_type = make_entity("Alice Smith", entity_type=EntityType.PERSON)
        diff_type = make_entity("Alice Smith", entity_type=EntityType.VESSEL)
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        assert resolver.score_observation(obs, same_type) > resolver.score_observation(obs, diff_type)

    def test_attribute_overlap_boosts_score(self):
        resolver = EntityResolver()
        with_attrs = make_entity("Alice Smith", attributes={"rank": "captain"})
        without_attrs = make_entity("Alice Smith")
        obs = make_obs("Alice Smith", attributes={"rank": "captain"}, source=INTSource.GEOINT)
        assert resolver.score_observation(obs, with_attrs) > resolver.score_observation(obs, without_attrs)

    def test_location_proximity_boosts_score(self):
        resolver = EntityResolver()
        near = make_entity("Alice Smith", location=(40.0, -73.0))
        far = make_entity("Alice Smith", location=(51.5, -0.12))
        obs = make_obs("Alice Smith", location=(40.001, -73.0), source=INTSource.GEOINT)
        assert resolver.score_observation(obs, near) > resolver.score_observation(obs, far)

    def test_source_reliability_weights_score(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        high = make_obs("Alice Smith", reliability=0.95)
        low = make_obs("Alice Smith", reliability=0.2)
        assert resolver.score_observation(high, entity) > resolver.score_observation(low, entity)

    def test_score_is_bounded_zero_to_one(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith", attributes={"rank": "captain"}, location=(40.0, -73.0))
        obs = make_obs(
            "Alice Smith",
            attributes={"rank": "captain"},
            location=(40.0, -73.0),
            source=INTSource.GEOINT,
            reliability=1.0,
        )
        score = resolver.score_observation(obs, entity)
        assert 0.0 <= score <= 1.0


class TestMatchObservation:
    def test_strong_match_returns_entity_and_score(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith", attributes={"rank": "captain"})
        obs = make_obs("Alice Smith", attributes={"rank": "captain"}, source=INTSource.GEOINT)
        match = resolver.match_observation(obs, [entity])
        assert match is not None
        assert match.entity is entity
        assert match.score > 0.5

    def test_no_match_below_threshold_returns_none(self):
        resolver = EntityResolver()
        entity = make_entity("Bob Jones")
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        assert resolver.match_observation(obs, [entity]) is None

    def test_empty_entity_list_returns_none(self):
        resolver = EntityResolver()
        obs = make_obs("Alice Smith")
        assert resolver.match_observation(obs, []) is None

    def test_best_candidate_wins(self):
        resolver = EntityResolver()
        alice = make_entity("Alice Smith")
        alicia = make_entity("Alicia Smith")
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        match = resolver.match_observation(obs, [alicia, alice])
        assert match is not None
        assert match.entity is alice


class TestDisambiguate:
    def test_disambiguates_among_candidates(self):
        resolver = EntityResolver()
        alice = make_entity("Alice Smith")
        alicia = make_entity("Alicia Smith")
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        ranked = resolver.disambiguate(obs, [alicia, alice])
        assert len(ranked) == 2
        assert ranked[0].entity is alice
        assert ranked[0].score >= ranked[1].score

    def test_disambiguate_empty_candidates_returns_empty(self):
        resolver = EntityResolver()
        obs = make_obs("Alice Smith")
        assert resolver.disambiguate(obs, []) == []


class TestResolve:
    def test_resolve_merges_observation_into_existing_entity(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        obs = make_obs("Alice Smith", attributes={"rank": "captain"}, source=INTSource.GEOINT)
        resolved = resolver.resolve(obs, [entity])
        assert resolved is entity
        assert obs in entity.observations
        assert "captain" in entity.attributes.get("rank", set())

    def test_resolve_creates_new_entity_when_no_match(self):
        resolver = EntityResolver()
        obs = make_obs("Alice Smith", source=INTSource.OSINT)
        resolved = resolver.resolve(obs, [])
        assert resolved is not None
        assert obs in resolved.observations
        assert "Alice Smith" in resolved.names

    def test_resolve_does_not_duplicate_observation(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        resolver.resolve(obs, [entity])
        resolver.resolve(obs, [entity])
        assert entity.observations.count(obs) == 1


class TestConfidence:
    def test_confidence_increases_with_source_count(self):
        resolver = EntityResolver()
        single = make_entity("Alice Smith")
        multi = make_entity("Alice Smith")
        multi.observations.append(make_obs("Alice Smith", source=INTSource.GEOINT))
        multi.observations.append(make_obs("Alice Smith", source=INTSource.SIGINT))
        assert resolver.confidence(multi) > resolver.confidence(single)

    def test_confidence_capped_at_one(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        for src in (INTSource.GEOINT, INTSource.SIGINT, INTSource.HUMINT, INTSource.MASINT):
            entity.observations.append(make_obs("Alice Smith", source=src))
        assert resolver.confidence(entity) <= 1.0

    def test_confidence_increases_with_observation_count(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        before = resolver.confidence(entity)
        entity.observations.append(make_obs("Alice Smith", source=INTSource.GEOINT))
        assert resolver.confidence(entity) > before

    def test_confidence_zero_for_entity_without_observations(self):
        resolver = EntityResolver()
        entity = ResolvedEntity(entity_id="e-x", entity_type=EntityType.PERSON)
        assert resolver.confidence(entity) == 0.0


class TestCrossIntResolution:
    def test_same_person_from_two_int_sources_resolves_to_one_entity(self):
        resolver = EntityResolver()
        osint_obs = make_obs("Alice Smith", source=INTSource.OSINT, attributes={"rank": "captain"})
        geoint_obs = make_obs("Alice Smith", source=INTSource.GEOINT, attributes={"rank": "captain"})
        first = resolver.resolve(osint_obs, [])
        second = resolver.resolve(geoint_obs, [first])
        assert second is first
        assert len(first.observations) == 2
        assert {o.source for o in first.observations} == {INTSource.OSINT, INTSource.GEOINT}

    def test_different_people_with_same_name_stay_separate(self):
        resolver = EntityResolver()
        alice_ny = make_obs("Alice Smith", source=INTSource.OSINT, location=(40.7, -74.0))
        alice_ldn = make_obs("Alice Smith", source=INTSource.OSINT, location=(51.5, -0.12))
        first = resolver.resolve(alice_ny, [])
        second = resolver.resolve(alice_ldn, [first])
        assert second is not first

    def test_resolve_merges_aliases(self):
        resolver = EntityResolver()
        obs1 = make_obs("Alice Smith", source=INTSource.OSINT)
        obs2 = make_obs("Alice Smith", aliases=("A. Smith",), source=INTSource.GEOINT)
        entity = resolver.resolve(obs1, [])
        entity = resolver.resolve(obs2, [entity])
        assert "A. Smith" in entity.aliases

    def test_confidence_higher_with_diverse_sources_than_single_source(self):
        resolver = EntityResolver()
        single = make_entity("Alice Smith")
        single.observations.append(make_obs("Alice Smith", source=INTSource.OSINT))
        single.observations.append(make_obs("Alice Smith", source=INTSource.OSINT))
        diverse = make_entity("Alice Smith")
        diverse.observations.append(make_obs("Alice Smith", source=INTSource.OSINT))
        diverse.observations.append(make_obs("Alice Smith", source=INTSource.GEOINT))
        assert resolver.confidence(diverse) > resolver.confidence(single)

    def test_match_result_carries_entity_and_score(self):
        resolver = EntityResolver()
        entity = make_entity("Alice Smith")
        obs = make_obs("Alice Smith", source=INTSource.GEOINT)
        match = resolver.match_observation(obs, [entity])
        assert match is not None
        assert match.entity is entity
        assert match.score == pytest.approx(resolver.score_observation(obs, entity))

    def test_resolve_type_mismatch_creates_new_entity(self):
        resolver = EntityResolver()
        person = make_entity("Alice Smith", entity_type=EntityType.PERSON)
        vessel_obs = make_obs("Alice Smith", entity_type=EntityType.VESSEL, source=INTSource.GEOINT)
        resolved = resolver.resolve(vessel_obs, [person])
        assert resolved is not person
        assert resolved.entity_type == EntityType.VESSEL
