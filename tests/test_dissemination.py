"""Tests for ISR Dissemination module — TDD enforced."""

from __future__ import annotations

import json
import time

import pytest

from src.isr.dissemination import (
    Classification,
    CoalitionPartner,
    CoalitionSharing,
    Priority,
    PriorityRouter,
    Product,
    ProductFormatter,
    ProductGenerator,
    ProductType,
)


# ── ProductGenerator tests ──────────────────────────────────────────


class TestProductGenerator:
    def test_generate_report_returns_product(self):
        gen = ProductGenerator()
        tracks = [
            {"track_id": "TRK-001", "position": (10.0, 20.0), "status": "confirmed"},
            {"track_id": "TRK-002", "position": (30.0, 40.0), "status": "tentative"},
        ]
        product = gen.generate_report(tracks, source="GEOINT")
        assert isinstance(product, Product)
        assert product.product_type == ProductType.REPORT
        assert product.source == "GEOINT"
        assert len(product.product_id) > 0

    def test_generate_report_includes_track_count(self):
        gen = ProductGenerator()
        tracks = [{"track_id": "TRK-001", "position": (1.0, 2.0)}]
        product = gen.generate_report(tracks)
        assert "1" in product.content

    def test_generate_alert_returns_product(self):
        gen = ProductGenerator()
        product = gen.generate_alert("Hostile aircraft detected", source="SIGINT")
        assert isinstance(product, Product)
        assert product.product_type == ProductType.ALERT
        assert product.priority == Priority.CRITICAL
        assert "Hostile aircraft detected" in product.content

    def test_generate_brief_returns_product(self):
        gen = ProductGenerator()
        product = gen.generate_brief("Daily ISR summary", source="HUMINT")
        assert isinstance(product, Product)
        assert product.product_type == ProductType.BRIEF

    def test_generate_summary_returns_product(self):
        gen = ProductGenerator()
        tracks = [{"track_id": f"TRK-{i:03d}"} for i in range(5)]
        product = gen.generate_summary(tracks)
        assert isinstance(product, Product)
        assert product.product_type == ProductType.SUMMARY

    def test_generate_report_with_empty_tracks(self):
        gen = ProductGenerator()
        product = gen.generate_report([])
        assert isinstance(product, Product)
        assert "0" in product.content


# ── ProductFormatter tests ──────────────────────────────────────────


class TestProductFormatter:
    def test_format_text_contains_title(self):
        product = Product(
            product_id="P-001",
            product_type=ProductType.REPORT,
            title="Test Report",
            content="Report body",
            priority=Priority.HIGH,
            classification=Classification.CONFIDENTIAL,
            source="GEOINT",
            timestamp=1000.0,
        )
        fmt = ProductFormatter()
        result = fmt.format_text(product)
        assert "Test Report" in result

    def test_format_text_contains_classification(self):
        product = Product(
            product_id="P-001",
            product_type=ProductType.REPORT,
            title="Test",
            content="Body",
            priority=Priority.LOW,
            classification=Classification.SECRET,
            source="SIGINT",
            timestamp=1000.0,
        )
        fmt = ProductFormatter()
        result = fmt.format_text(product)
        assert "SECRET" in result

    def test_format_json_returns_valid_json(self):
        product = Product(
            product_id="P-001",
            product_type=ProductType.ALERT,
            title="Alert",
            content="Alert body",
            priority=Priority.CRITICAL,
            classification=Classification.TOP_SECRET,
            source="HUMINT",
            timestamp=2000.0,
        )
        fmt = ProductFormatter()
        result = fmt.format_json(product)
        data = json.loads(result)
        assert data["product_id"] == "P-001"
        assert data["title"] == "Alert"

    def test_format_xml_contains_product_id(self):
        product = Product(
            product_id="P-999",
            product_type=ProductType.BRIEF,
            title="Brief",
            content="Brief body",
            priority=Priority.MEDIUM,
            classification=Classification.UNCLASSIFIED,
            source="OSINT",
            timestamp=3000.0,
        )
        fmt = ProductFormatter()
        result = fmt.format_xml(product)
        assert "P-999" in result


# ── PriorityRouter tests ────────────────────────────────────────────


class TestPriorityRouter:
    def test_route_critical_to_immediate(self):
        router = PriorityRouter()
        product = Product(
            product_id="P-001",
            product_type=ProductType.ALERT,
            title="Critical Alert",
            content="Body",
            priority=Priority.CRITICAL,
            classification=Classification.SECRET,
            source="SIGINT",
            timestamp=1000.0,
        )
        channel = router.route(product)
        assert channel == "immediate"

    def test_route_high_to_priority(self):
        router = PriorityRouter()
        product = Product(
            product_id="P-002",
            product_type=ProductType.REPORT,
            title="High Report",
            content="Body",
            priority=Priority.HIGH,
            classification=Classification.CONFIDENTIAL,
            source="GEOINT",
            timestamp=1000.0,
        )
        channel = router.route(product)
        assert channel == "priority"

    def test_route_medium_to_routine(self):
        router = PriorityRouter()
        product = Product(
            product_id="P-003",
            product_type=ProductType.BRIEF,
            title="Medium Brief",
            content="Body",
            priority=Priority.MEDIUM,
            classification=Classification.UNCLASSIFIED,
            source="OSINT",
            timestamp=1000.0,
        )
        channel = router.route(product)
        assert channel == "routine"

    def test_route_low_to_deferred(self):
        router = PriorityRouter()
        product = Product(
            product_id="P-004",
            product_type=ProductType.SUMMARY,
            title="Low Summary",
            content="Body",
            priority=Priority.LOW,
            classification=Classification.UNCLASSIFIED,
            source="HUMINT",
            timestamp=1000.0,
        )
        channel = router.route(product)
        assert channel == "deferred"

    def test_route_all_groups_by_channel(self):
        router = PriorityRouter()
        products = [
            Product(
                product_id=f"P-{i:03d}",
                product_type=ProductType.REPORT,
                title=f"Report {i}",
                content="Body",
                priority=prio,
                classification=Classification.UNCLASSIFIED,
                source="GEOINT",
                timestamp=1000.0,
            )
            for i, prio in enumerate(
                [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]
            )
        ]
        result = router.route_all(products)
        assert "immediate" in result
        assert "priority" in result
        assert "routine" in result
        assert "deferred" in result
        assert len(result["immediate"]) == 1
        assert len(result["priority"]) == 1


# ── CoalitionSharing tests ──────────────────────────────────────────


class TestCoalitionSharing:
    def test_can_share_unclassified_with_nato(self):
        partner = CoalitionPartner(name="NATO", clearance=Classification.SECRET)
        sharing = CoalitionSharing()
        product = Product(
            product_id="P-001",
            product_type=ProductType.REPORT,
            title="Report",
            content="Body",
            priority=Priority.LOW,
            classification=Classification.UNCLASSIFIED,
            source="OSINT",
            timestamp=1000.0,
        )
        assert sharing.can_share(product, partner) is True

    def test_cannot_share_secret_with_unclassified_partner(self):
        partner = CoalitionPartner(name="PartnerX", clearance=Classification.UNCLASSIFIED)
        sharing = CoalitionSharing()
        product = Product(
            product_id="P-002",
            product_type=ProductType.ALERT,
            title="Alert",
            content="Body",
            priority=Priority.HIGH,
            classification=Classification.SECRET,
            source="SIGINT",
            timestamp=1000.0,
        )
        assert sharing.can_share(product, partner) is False

    def test_can_share_secret_with_top_secret_partner(self):
        partner = CoalitionPartner(name="FiveEyes", clearance=Classification.TOP_SECRET)
        sharing = CoalitionSharing()
        product = Product(
            product_id="P-003",
            product_type=ProductType.REPORT,
            title="Report",
            content="Body",
            priority=Priority.MEDIUM,
            classification=Classification.SECRET,
            source="GEOINT",
            timestamp=1000.0,
        )
        assert sharing.can_share(product, partner) is True

    def test_share_returns_share_record(self):
        partner = CoalitionPartner(name="NATO", clearance=Classification.SECRET)
        sharing = CoalitionSharing()
        product = Product(
            product_id="P-004",
            product_type=ProductType.BRIEF,
            title="Brief",
            content="Body",
            priority=Priority.MEDIUM,
            classification=Classification.CONFIDENTIAL,
            source="HUMINT",
            timestamp=1000.0,
        )
        record = sharing.share(product, partner)
        assert record["product_id"] == "P-004"
        assert record["partner"] == "NATO"
        assert "timestamp" in record

    def test_share_denied_raises_for_incompatible(self):
        partner = CoalitionPartner(name="LowClearance", clearance=Classification.UNCLASSIFIED)
        sharing = CoalitionSharing()
        product = Product(
            product_id="P-005",
            product_type=ProductType.ALERT,
            title="Alert",
            content="Body",
            priority=Priority.CRITICAL,
            classification=Classification.TOP_SECRET,
            source="SIGINT",
            timestamp=1000.0,
        )
        with pytest.raises(PermissionError):
            sharing.share(product, partner)

    def test_get_shareable_products_filters_correctly(self):
        partner = CoalitionPartner(name="NATO", clearance=Classification.SECRET)
        sharing = CoalitionSharing()
        products = [
            Product(
                product_id="P-001",
                product_type=ProductType.REPORT,
                title="Unclass",
                content="Body",
                priority=Priority.LOW,
                classification=Classification.UNCLASSIFIED,
                source="OSINT",
                timestamp=1000.0,
            ),
            Product(
                product_id="P-002",
                product_type=ProductType.ALERT,
                title="Secret",
                content="Body",
                priority=Priority.HIGH,
                classification=Classification.SECRET,
                source="SIGINT",
                timestamp=1000.0,
            ),
            Product(
                product_id="P-003",
                product_type=ProductType.BRIEF,
                title="TopSecret",
                content="Body",
                priority=Priority.CRITICAL,
                classification=Classification.TOP_SECRET,
                source="HUMINT",
                timestamp=1000.0,
            ),
        ]
        shareable = sharing.get_shareable_products(products, partner)
        ids = [p.product_id for p in shareable]
        assert "P-001" in ids
        assert "P-002" in ids
        assert "P-003" not in ids
