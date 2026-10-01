"""ISR Dissemination — product generation, formatting, priority routing, and coalition sharing."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Classification(Enum):
    """Security classification levels."""

    UNCLASSIFIED = 0
    CONFIDENTIAL = 1
    SECRET = 2
    TOP_SECRET = 3


class Priority(Enum):
    """Product priority levels."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class ProductType(Enum):
    """Types of ISR products."""

    REPORT = "report"
    ALERT = "alert"
    BRIEF = "brief"
    SUMMARY = "summary"


@dataclass
class Product:
    """An ISR product ready for dissemination."""

    product_id: str
    product_type: ProductType
    title: str
    content: str
    priority: Priority
    classification: Classification
    source: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CoalitionPartner:
    """A coalition partner with a security clearance."""

    name: str
    clearance: Classification
    shareable_product_types: list[ProductType] = field(default_factory=list)


class ProductGenerator:
    """Generates ISR products from tracks and intelligence."""

    def generate_report(
        self,
        tracks: list[dict[str, Any]],
        source: str = "GEOINT",
        title: str | None = None,
    ) -> Product:
        """Generate a report product from a list of tracks."""
        track_count = len(tracks)
        track_ids = [t.get("track_id", "UNKNOWN") for t in tracks]
        content_lines = [f"ISR Report — {source}", f"Track count: {track_count}", ""]
        for tid in track_ids:
            content_lines.append(f"  - {tid}")
        content = "\n".join(content_lines)
        return Product(
            product_id=f"RPT-{uuid.uuid4().hex[:8].upper()}",
            product_type=ProductType.REPORT,
            title=title or f"ISR Report ({source})",
            content=content,
            priority=Priority.MEDIUM,
            classification=Classification.CONFIDENTIAL,
            source=source,
            timestamp=time.time(),
        )

    def generate_alert(
        self,
        message: str,
        source: str = "SIGINT",
        title: str | None = None,
    ) -> Product:
        """Generate an alert product."""
        return Product(
            product_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
            product_type=ProductType.ALERT,
            title=title or "ISR Alert",
            content=message,
            priority=Priority.CRITICAL,
            classification=Classification.SECRET,
            source=source,
            timestamp=time.time(),
        )

    def generate_brief(
        self,
        summary: str,
        source: str = "HUMINT",
        title: str | None = None,
    ) -> Product:
        """Generate a brief product."""
        return Product(
            product_id=f"BRF-{uuid.uuid4().hex[:8].upper()}",
            product_type=ProductType.BRIEF,
            title=title or "ISR Brief",
            content=summary,
            priority=Priority.MEDIUM,
            classification=Classification.CONFIDENTIAL,
            source=source,
            timestamp=time.time(),
        )

    def generate_summary(
        self,
        tracks: list[dict[str, Any]],
        source: str = "GEOINT",
        title: str | None = None,
    ) -> Product:
        """Generate a summary product from tracks."""
        track_count = len(tracks)
        content = f"ISR Summary — {source}\nTotal tracks: {track_count}"
        return Product(
            product_id=f"SUM-{uuid.uuid4().hex[:8].upper()}",
            product_type=ProductType.SUMMARY,
            title=title or f"ISR Summary ({source})",
            content=content,
            priority=Priority.LOW,
            classification=Classification.UNCLASSIFIED,
            source=source,
            timestamp=time.time(),
        )


class ProductFormatter:
    """Formats ISR products for dissemination."""

    def format_text(self, product: Product) -> str:
        """Format product as plain text."""
        lines = [
            f"[{product.classification.name}] {product.title}",
            f"Type: {product.product_type.value}",
            f"Priority: {product.priority.name}",
            f"Source: {product.source}",
            f"ID: {product.product_id}",
            "",
            product.content,
        ]
        return "\n".join(lines)

    def format_json(self, product: Product) -> str:
        """Format product as JSON."""
        data = {
            "product_id": product.product_id,
            "product_type": product.product_type.value,
            "title": product.title,
            "content": product.content,
            "priority": product.priority.name,
            "classification": product.classification.name,
            "source": product.source,
            "timestamp": product.timestamp,
            "metadata": product.metadata,
        }
        return json.dumps(data, indent=2)

    def format_xml(self, product: Product) -> str:
        """Format product as XML."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            "<product>",
            f"  <product_id>{product.product_id}</product_id>",
            f"  <product_type>{product.product_type.value}</product_type>",
            f"  <title>{product.title}</title>",
            f"  <content>{product.content}</content>",
            f"  <priority>{product.priority.name}</priority>",
            f"  <classification>{product.classification.name}</classification>",
            f"  <source>{product.source}</source>",
            f"  <timestamp>{product.timestamp}</timestamp>",
            "</product>",
        ]
        return "\n".join(lines)


class PriorityRouter:
    """Routes ISR products to dissemination channels based on priority."""

    CHANNEL_MAP = {
        Priority.CRITICAL: "immediate",
        Priority.HIGH: "priority",
        Priority.MEDIUM: "routine",
        Priority.LOW: "deferred",
    }

    def route(self, product: Product) -> str:
        """Route a single product to a channel."""
        return self.CHANNEL_MAP[product.priority]

    def route_all(self, products: list[Product]) -> dict[str, list[Product]]:
        """Route all products, grouped by channel."""
        result: dict[str, list[Product]] = {}
        for product in products:
            channel = self.route(product)
            result.setdefault(channel, []).append(product)
        return result


class CoalitionSharing:
    """Manages sharing of ISR products with coalition partners."""

    def can_share(self, product: Product, partner: CoalitionPartner) -> bool:
        """Check if a product can be shared with a partner."""
        return product.classification.value <= partner.clearance.value

    def share(
        self, product: Product, partner: CoalitionPartner
    ) -> dict[str, Any]:
        """Share a product with a partner. Raises PermissionError if not allowed."""
        if not self.can_share(product, partner):
            raise PermissionError(
                f"Cannot share {product.classification.name} product "
                f"with {partner.name} (clearance: {partner.clearance.name})"
            )
        return {
            "product_id": product.product_id,
            "partner": partner.name,
            "timestamp": time.time(),
            "classification": product.classification.name,
        }

    def get_shareable_products(
        self, products: list[Product], partner: CoalitionPartner
    ) -> list[Product]:
        """Filter products to only those shareable with a partner."""
        return [p for p in products if self.can_share(p, partner)]
