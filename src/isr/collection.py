"""ISR Collection Management — sensor tasking, collection planning, priority management, resource allocation.

This module implements:
- Sensor tasking: assign collection tasks to sensors based on capability and availability
- Collection planning: create and manage collection plans with multiple tasks
- Priority management: task prioritization and preemption
- Resource allocation: sensor capacity management and conflict resolution
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Priority(Enum):
    """Task priority levels (lower value = higher priority)."""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    ROUTINE = 5


class SensorType(Enum):
    """Types of ISR sensors."""

    RADAR = "radar"
    EO_IR = "eo_ir"
    SIGINT = "sigint"
    SAR = "sar"
    HUMINT = "humint"
    GEOINT = "geoint"


class SensorStatus(Enum):
    """Operational status of a sensor."""

    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class TaskStatus(Enum):
    """Lifecycle status of a collection task."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PREEMPTED = "preempted"


@dataclass
class Sensor:
    """An ISR sensor with capabilities and capacity."""

    sensor_id: str
    sensor_type: SensorType
    status: SensorStatus = SensorStatus.IDLE
    max_concurrent: int = 1
    capabilities: list[str] = field(default_factory=list)
    current_tasks: list[str] = field(default_factory=list)
    location: tuple[float, float] = (0.0, 0.0)

    @property
    def is_available(self) -> bool:
        """Check if sensor can accept new tasks."""
        return self.status == SensorStatus.IDLE and len(self.current_tasks) < self.max_concurrent

    @property
    def available_capacity(self) -> int:
        """Remaining task capacity."""
        return self.max_concurrent - len(self.current_tasks)


@dataclass
class CollectionTask:
    """A single collection task."""

    task_id: str
    sensor_type: SensorType
    priority: Priority
    target_id: str
    duration: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    assigned_sensor: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CollectionPlan:
    """A collection plan containing multiple tasks."""

    plan_id: str
    tasks: list[CollectionTask] = field(default_factory=list)
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pending_tasks(self) -> list[CollectionTask]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    @property
    def completed_tasks(self) -> list[CollectionTask]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    @property
    def is_complete(self) -> bool:
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks) and len(self.tasks) > 0


class ISRCollectionManager:
    """Manages ISR sensor tasking, planning, priorities, and resource allocation."""

    def __init__(self):
        self._sensors: dict[str, Sensor] = {}
        self._tasks: dict[str, CollectionTask] = {}
        self._plans: dict[str, CollectionPlan] = {}
        self._lock = threading.RLock()
        self._next_task_id: int = 1
        self._next_plan_id: int = 1

    def register_sensor(self, sensor: Sensor) -> None:
        """Register a sensor for tasking."""
        with self._lock:
            self._sensors[sensor.sensor_id] = sensor

    def unregister_sensor(self, sensor_id: str) -> bool:
        """Remove a sensor from the manager."""
        with self._lock:
            if sensor_id not in self._sensors:
                return False
            del self._sensors[sensor_id]
            return True

    def get_sensor(self, sensor_id: str) -> Optional[Sensor]:
        """Get a sensor by ID."""
        return self._sensors.get(sensor_id)

    def get_all_sensors(self) -> list[Sensor]:
        """Get all registered sensors."""
        return list(self._sensors.values())

    def get_available_sensors(self, sensor_type: Optional[SensorType] = None) -> list[Sensor]:
        """Get available sensors, optionally filtered by type."""
        sensors = [s for s in self._sensors.values() if s.is_available]
        if sensor_type is not None:
            sensors = [s for s in sensors if s.sensor_type == sensor_type]
        return sensors

    def create_task(
        self,
        sensor_type: SensorType,
        target_id: str,
        priority: Priority = Priority.MEDIUM,
        duration: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CollectionTask:
        """Create a new collection task."""
        task_id = f"TASK-{self._next_task_id:06d}"
        self._next_task_id += 1
        task = CollectionTask(
            task_id=task_id,
            sensor_type=sensor_type,
            priority=priority,
            target_id=target_id,
            duration=duration,
            created_at=time.time(),
            metadata=metadata or {},
        )
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[CollectionTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[CollectionTask]:
        """Get all tasks."""
        return list(self._tasks.values())

    def get_tasks_by_status(self, status: TaskStatus) -> list[CollectionTask]:
        """Get tasks filtered by status."""
        return [t for t in self._tasks.values() if t.status == status]

    def get_tasks_by_priority(self, priority: Priority) -> list[CollectionTask]:
        """Get tasks filtered by priority."""
        return [t for t in self._tasks.values() if t.priority == priority]

    def create_plan(self, plan_id: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> CollectionPlan:
        """Create a new collection plan."""
        if plan_id is None:
            plan_id = f"PLAN-{self._next_plan_id:06d}"
            self._next_plan_id += 1
        plan = CollectionPlan(plan_id=plan_id, created_at=time.time(), metadata=metadata or {})
        with self._lock:
            self._plans[plan_id] = plan
        return plan

    def get_plan(self, plan_id: str) -> Optional[CollectionPlan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def add_task_to_plan(self, plan_id: str, task: CollectionTask) -> bool:
        """Add a task to a plan."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                return False
            plan.tasks.append(task)
            return True

    def assign_task(self, task_id: str, sensor_id: str) -> bool:
        """Assign a task to a specific sensor."""
        with self._lock:
            task = self._tasks.get(task_id)
            sensor = self._sensors.get(sensor_id)
            if task is None or sensor is None:
                return False
            if task.status not in (TaskStatus.PENDING, TaskStatus.PREEMPTED):
                return False
            if not sensor.is_available:
                return False
            if sensor.sensor_type != task.sensor_type:
                return False
            task.assigned_sensor = sensor_id
            task.status = TaskStatus.ASSIGNED
            sensor.current_tasks.append(task_id)
            if len(sensor.current_tasks) >= sensor.max_concurrent:
                sensor.status = SensorStatus.BUSY
            return True

    def auto_assign_task(self, task_id: str) -> Optional[str]:
        """Automatically assign a task to the best available sensor."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status not in (TaskStatus.PENDING, TaskStatus.PREEMPTED):
                return None
            candidates = self.get_available_sensors(task.sensor_type)
            if not candidates:
                return None
            # Pick sensor with most available capacity
            best = max(candidates, key=lambda s: s.available_capacity)
            if self.assign_task(task_id, best.sensor_id):
                return best.sensor_id
            return None

    def start_task(self, task_id: str) -> bool:
        """Mark a task as in progress."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.ASSIGNED:
                return False
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = time.time()
            return True

    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed and free sensor capacity."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.IN_PROGRESS:
                return False
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            if task.assigned_sensor:
                sensor = self._sensors.get(task.assigned_sensor)
                if sensor and task_id in sensor.current_tasks:
                    sensor.current_tasks.remove(task_id)
                    if sensor.status == SensorStatus.BUSY and len(sensor.current_tasks) < sensor.max_concurrent:
                        sensor.status = SensorStatus.IDLE
            return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or assigned task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status not in (TaskStatus.PENDING, TaskStatus.ASSIGNED):
                return False
            if task.assigned_sensor:
                sensor = self._sensors.get(task.assigned_sensor)
                if sensor and task_id in sensor.current_tasks:
                    sensor.current_tasks.remove(task_id)
                    if sensor.status == SensorStatus.BUSY and len(sensor.current_tasks) < sensor.max_concurrent:
                        sensor.status = SensorStatus.IDLE
            task.status = TaskStatus.CANCELLED
            return True

    def preempt_task(self, task_id: str) -> bool:
        """Preempt an in-progress task (for higher priority tasking)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != TaskStatus.IN_PROGRESS:
                return False
            task.status = TaskStatus.PREEMPTED
            if task.assigned_sensor:
                sensor = self._sensors.get(task.assigned_sensor)
                if sensor and task_id in sensor.current_tasks:
                    sensor.current_tasks.remove(task_id)
                    if sensor.status == SensorStatus.BUSY and len(sensor.current_tasks) < sensor.max_concurrent:
                        sensor.status = SensorStatus.IDLE
            return True

    def get_pending_tasks_sorted(self) -> list[CollectionTask]:
        """Get pending tasks sorted by priority (highest first)."""
        pending = [t for t in self._tasks.values() if t.status == TaskStatus.PENDING]
        return sorted(pending, key=lambda t: (t.priority.value, t.created_at))

    def allocate_plan(self, plan_id: str) -> dict[str, Any]:
        """Allocate all pending tasks in a plan to available sensors."""
        with self._lock:
            plan = self._plans.get(plan_id)
            if plan is None:
                return {"assigned": 0, "unassigned": 0, "details": []}
            pending = [t for t in plan.tasks if t.status == TaskStatus.PENDING]
            pending.sort(key=lambda t: (t.priority.value, t.created_at))
            assigned = 0
            unassigned = 0
            details = []
            for task in pending:
                sensor_id = self.auto_assign_task(task.task_id)
                if sensor_id:
                    assigned += 1
                    details.append({"task_id": task.task_id, "sensor_id": sensor_id})
                else:
                    unassigned += 1
                    details.append({"task_id": task.task_id, "sensor_id": None})
            return {"assigned": assigned, "unassigned": unassigned, "details": details}

    def get_sensor_utilization(self) -> dict[str, Any]:
        """Get utilization statistics for all sensors."""
        sensors = list(self._sensors.values())
        total_capacity = sum(s.max_concurrent for s in sensors)
        used_capacity = sum(len(s.current_tasks) for s in sensors)
        return {
            "total_sensors": len(sensors),
            "total_capacity": total_capacity,
            "used_capacity": used_capacity,
            "utilization": used_capacity / total_capacity if total_capacity > 0 else 0.0,
            "by_status": {
                status.value: sum(1 for s in sensors if s.status == status)
                for status in SensorStatus
            },
        }

    def get_stats(self) -> dict[str, Any]:
        """Get overall collection management statistics."""
        tasks = list(self._tasks.values())
        return {
            "total_tasks": len(tasks),
            "total_sensors": len(self._sensors),
            "total_plans": len(self._plans),
            "by_task_status": {
                status.value: sum(1 for t in tasks if t.status == status)
                for status in TaskStatus
            },
            "by_priority": {
                priority.name: sum(1 for t in tasks if t.priority == priority)
                for priority in Priority
            },
        }

    def reset(self) -> None:
        """Clear all state."""
        with self._lock:
            self._sensors.clear()
            self._tasks.clear()
            self._plans.clear()
            self._next_task_id = 1
            self._next_plan_id = 1
