"""Tests for ISR collection management — sensor tasking, planning, priorities, allocation."""

import pytest

from src.isr.collection import (
    CollectionPlan,
    CollectionTask,
    ISRCollectionManager,
    Priority,
    Sensor,
    SensorStatus,
    SensorType,
    TaskStatus,
)


class TestEnumsAndSensor:
    def test_priority_ordering(self):
        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.MEDIUM.value
        assert Priority.MEDIUM.value < Priority.LOW.value
        assert Priority.LOW.value < Priority.ROUTINE.value

    def test_sensor_type_values(self):
        assert SensorType.RADAR.value == "radar"
        assert SensorType.EO_IR.value == "eo_ir"
        assert SensorType.SIGINT.value == "sigint"
        assert SensorType.SAR.value == "sar"

    def test_sensor_creation_defaults(self):
        sensor = Sensor(sensor_id="S1", sensor_type=SensorType.RADAR)
        assert sensor.status == SensorStatus.IDLE
        assert sensor.max_concurrent == 1
        assert sensor.available_capacity == 1
        assert sensor.is_available

    def test_sensor_unavailable_when_offline(self):
        sensor = Sensor(sensor_id="S1", sensor_type=SensorType.RADAR, status=SensorStatus.OFFLINE)
        assert not sensor.is_available
        assert sensor.available_capacity == 1


class TestTaskLifecycle:
    def test_create_task_defaults(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert task.status == TaskStatus.PENDING
        assert task.priority == Priority.MEDIUM
        assert task.assigned_sensor is None
        assert task.task_id.startswith("TASK-")

    def test_task_ids_are_unique(self):
        mgr = ISRCollectionManager()
        t1 = mgr.create_task(SensorType.RADAR, "TGT-1")
        t2 = mgr.create_task(SensorType.RADAR, "TGT-2")
        assert t1.task_id != t2.task_id

    def test_get_task(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.SIGINT, "TGT-9", priority=Priority.HIGH)
        assert mgr.get_task(task.task_id) is task
        assert mgr.get_task("NONEXISTENT") is None

    def test_get_tasks_by_status(self):
        mgr = ISRCollectionManager()
        t1 = mgr.create_task(SensorType.RADAR, "A")
        t2 = mgr.create_task(SensorType.RADAR, "B")
        mgr.cancel_task(t2.task_id)
        assert mgr.get_tasks_by_status(TaskStatus.PENDING) == [t1]
        assert mgr.get_tasks_by_status(TaskStatus.CANCELLED) == [t2]

    def test_get_tasks_by_priority(self):
        mgr = ISRCollectionManager()
        t1 = mgr.create_task(SensorType.RADAR, "A", priority=Priority.LOW)
        t2 = mgr.create_task(SensorType.RADAR, "B", priority=Priority.CRITICAL)
        t3 = mgr.create_task(SensorType.RADAR, "C", priority=Priority.CRITICAL)
        assert mgr.get_tasks_by_priority(Priority.CRITICAL) == [t2, t3]
        assert mgr.get_tasks_by_priority(Priority.LOW) == [t1]

    def test_cancel_pending_task(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.RADAR, "A")
        assert mgr.cancel_task(task.task_id)
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent_task(self):
        mgr = ISRCollectionManager()
        assert not mgr.cancel_task("NOPE")

    def test_cancel_completed_task_fails(self):
        mgr = ISRCollectionManager()
        sensor = Sensor(sensor_id="S1", sensor_type=SensorType.RADAR)
        mgr.register_sensor(sensor)
        task = mgr.create_task(SensorType.RADAR, "A")
        mgr.auto_assign_task(task.task_id)
        mgr.start_task(task.task_id)
        mgr.complete_task(task.task_id)
        assert not mgr.cancel_task(task.task_id)
        assert task.status == TaskStatus.COMPLETED


class TestSensorRegistration:
    def test_register_and_get_sensor(self):
        mgr = ISRCollectionManager()
        sensor = Sensor(sensor_id="S1", sensor_type=SensorType.RADAR)
        mgr.register_sensor(sensor)
        assert mgr.get_sensor("S1") is sensor
        assert mgr.get_all_sensors() == [sensor]

    def test_unregister_sensor(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        assert mgr.unregister_sensor("S1")
        assert mgr.get_sensor("S1") is None
        assert not mgr.unregister_sensor("S1")

    def test_get_available_sensors_filtered_by_type(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="R1", sensor_type=SensorType.RADAR))
        mgr.register_sensor(Sensor(sensor_id="E1", sensor_type=SensorType.EO_IR))
        radar_sensors = mgr.get_available_sensors(SensorType.RADAR)
        assert [s.sensor_id for s in radar_sensors] == ["R1"]
        all_sensors = mgr.get_available_sensors()
        assert len(all_sensors) == 2

    def test_offline_sensor_not_available(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR, status=SensorStatus.OFFLINE))
        assert mgr.get_available_sensors() == []


class TestTaskAssignment:
    def test_assign_task_to_matching_sensor(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert mgr.assign_task(task.task_id, "S1")
        assert task.status == TaskStatus.ASSIGNED
        assert task.assigned_sensor == "S1"

    def test_assign_task_wrong_sensor_type_fails(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.EO_IR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert not mgr.assign_task(task.task_id, "S1")
        assert task.status == TaskStatus.PENDING

    def test_assign_task_no_sensor_fails(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert not mgr.assign_task(task.task_id, "NOPE")

    def test_assign_already_assigned_task_fails(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        mgr.register_sensor(Sensor(sensor_id="S2", sensor_type=SensorType.RADAR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert mgr.assign_task(task.task_id, "S1")
        assert not mgr.assign_task(task.task_id, "S2")

    def test_auto_assign_task(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        sensor_id = mgr.auto_assign_task(task.task_id)
        assert sensor_id == "S1"
        assert task.status == TaskStatus.ASSIGNED

    def test_auto_assign_no_matching_sensor(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.EO_IR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert mgr.auto_assign_task(task.task_id) is None
        assert task.status == TaskStatus.PENDING

    def test_auto_assign_picks_highest_capacity(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR, max_concurrent=1))
        mgr.register_sensor(Sensor(sensor_id="S2", sensor_type=SensorType.RADAR, max_concurrent=3))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert mgr.auto_assign_task(task.task_id) == "S2"

    def test_start_task(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        mgr.auto_assign_task(task.task_id)
        assert mgr.start_task(task.task_id)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None

    def test_start_unassigned_task_fails(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert not mgr.start_task(task.task_id)

    def test_complete_task_frees_sensor(self):
        mgr = ISRCollectionManager()
        sensor = Sensor(sensor_id="S1", sensor_type=SensorType.RADAR)
        mgr.register_sensor(sensor)
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        mgr.auto_assign_task(task.task_id)
        mgr.start_task(task.task_id)
        assert mgr.complete_task(task.task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert sensor.status == SensorStatus.IDLE
        assert sensor.current_tasks == []

    def test_complete_unstarted_task_fails(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert not mgr.complete_task(task.task_id)


class TestPreemption:
    def test_preempt_in_progress_task(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        mgr.auto_assign_task(task.task_id)
        mgr.start_task(task.task_id)
        assert mgr.preempt_task(task.task_id)
        assert task.status == TaskStatus.PREEMPTED

    def test_preempt_pending_task_fails(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert not mgr.preempt_task(task.task_id)
        assert task.status == TaskStatus.PENDING

    def test_preempt_frees_sensor_for_reassignment(self):
        mgr = ISRCollectionManager()
        sensor = Sensor(sensor_id="S1", sensor_type=SensorType.RADAR)
        mgr.register_sensor(sensor)
        t1 = mgr.create_task(SensorType.RADAR, "TGT-1")
        mgr.auto_assign_task(t1.task_id)
        mgr.start_task(t1.task_id)
        mgr.preempt_task(t1.task_id)
        t2 = mgr.create_task(SensorType.RADAR, "TGT-2", priority=Priority.CRITICAL)
        assert mgr.auto_assign_task(t2.task_id) == "S1"
        assert t2.status == TaskStatus.ASSIGNED

    def test_reassign_preempted_task(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        mgr.auto_assign_task(task.task_id)
        mgr.start_task(task.task_id)
        mgr.preempt_task(task.task_id)
        assert mgr.auto_assign_task(task.task_id) == "S1"
        assert task.status == TaskStatus.ASSIGNED


class TestPriorityManagement:
    def test_pending_tasks_sorted_by_priority(self):
        mgr = ISRCollectionManager()
        t1 = mgr.create_task(SensorType.RADAR, "A", priority=Priority.LOW)
        t2 = mgr.create_task(SensorType.RADAR, "B", priority=Priority.CRITICAL)
        t3 = mgr.create_task(SensorType.RADAR, "C", priority=Priority.HIGH)
        sorted_tasks = mgr.get_pending_tasks_sorted()
        assert [t.task_id for t in sorted_tasks] == [t2.task_id, t3.task_id, t1.task_id]

    def test_pending_tasks_same_priority_sorted_by_creation(self):
        mgr = ISRCollectionManager()
        t1 = mgr.create_task(SensorType.RADAR, "A", priority=Priority.MEDIUM)
        t2 = mgr.create_task(SensorType.RADAR, "B", priority=Priority.MEDIUM)
        sorted_tasks = mgr.get_pending_tasks_sorted()
        assert [t.task_id for t in sorted_tasks] == [t1.task_id, t2.task_id]

    def test_get_pending_tasks_sorted_excludes_non_pending(self):
        mgr = ISRCollectionManager()
        t1 = mgr.create_task(SensorType.RADAR, "A", priority=Priority.LOW)
        t2 = mgr.create_task(SensorType.RADAR, "B", priority=Priority.CRITICAL)
        mgr.cancel_task(t2.task_id)
        sorted_tasks = mgr.get_pending_tasks_sorted()
        assert sorted_tasks == [t1]


class TestCollectionPlans:
    def test_create_plan(self):
        mgr = ISRCollectionManager()
        plan = mgr.create_plan()
        assert plan.plan_id.startswith("PLAN-")
        assert plan.tasks == []
        assert mgr.get_plan(plan.plan_id) is plan

    def test_create_plan_with_custom_id(self):
        mgr = ISRCollectionManager()
        plan = mgr.create_plan(plan_id="MY-PLAN")
        assert plan.plan_id == "MY-PLAN"

    def test_add_task_to_plan(self):
        mgr = ISRCollectionManager()
        plan = mgr.create_plan()
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert mgr.add_task_to_plan(plan.plan_id, task)
        assert plan.tasks == [task]

    def test_add_task_to_nonexistent_plan(self):
        mgr = ISRCollectionManager()
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        assert not mgr.add_task_to_plan("NOPE", task)

    def test_plan_is_complete(self):
        mgr = ISRCollectionManager()
        plan = mgr.create_plan()
        assert not plan.is_complete
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        mgr.add_task_to_plan(plan.plan_id, task)
        assert not plan.is_complete
        task.status = TaskStatus.COMPLETED
        assert plan.is_complete

    def test_plan_pending_and_completed(self):
        mgr = ISRCollectionManager()
        plan = mgr.create_plan()
        t1 = mgr.create_task(SensorType.RADAR, "A")
        t2 = mgr.create_task(SensorType.RADAR, "B")
        mgr.add_task_to_plan(plan.plan_id, t1)
        mgr.add_task_to_plan(plan.plan_id, t2)
        t1.status = TaskStatus.COMPLETED
        assert plan.pending_tasks == [t2]
        assert plan.completed_tasks == [t1]


class TestResourceAllocation:
    def test_allocate_plan(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        plan = mgr.create_plan()
        t1 = mgr.create_task(SensorType.RADAR, "A", priority=Priority.HIGH)
        t2 = mgr.create_task(SensorType.RADAR, "B", priority=Priority.LOW)
        mgr.add_task_to_plan(plan.plan_id, t1)
        mgr.add_task_to_plan(plan.plan_id, t2)
        result = mgr.allocate_plan(plan.plan_id)
        assert result["assigned"] == 1
        assert result["unassigned"] == 1
        assert result["details"][0]["task_id"] == t1.task_id
        assert result["details"][0]["sensor_id"] == "S1"

    def test_allocate_plan_respects_priority(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR, max_concurrent=1))
        plan = mgr.create_plan()
        t1 = mgr.create_task(SensorType.RADAR, "A", priority=Priority.LOW)
        t2 = mgr.create_task(SensorType.RADAR, "B", priority=Priority.CRITICAL)
        mgr.add_task_to_plan(plan.plan_id, t1)
        mgr.add_task_to_plan(plan.plan_id, t2)
        result = mgr.allocate_plan(plan.plan_id)
        assert result["assigned"] == 1
        assert result["details"][0]["task_id"] == t2.task_id

    def test_allocate_nonexistent_plan(self):
        mgr = ISRCollectionManager()
        result = mgr.allocate_plan("NOPE")
        assert result == {"assigned": 0, "unassigned": 0, "details": []}

    def test_sensor_utilization(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR, max_concurrent=2))
        mgr.register_sensor(Sensor(sensor_id="S2", sensor_type=SensorType.EO_IR, max_concurrent=1))
        util = mgr.get_sensor_utilization()
        assert util["total_sensors"] == 2
        assert util["total_capacity"] == 3
        assert util["used_capacity"] == 0
        assert util["utilization"] == 0.0

    def test_sensor_utilization_with_active_tasks(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR, max_concurrent=2))
        task = mgr.create_task(SensorType.RADAR, "TGT-1")
        mgr.auto_assign_task(task.task_id)
        util = mgr.get_sensor_utilization()
        assert util["used_capacity"] == 1
        assert util["utilization"] == pytest.approx(0.5)

    def test_sensor_utilization_no_sensors(self):
        mgr = ISRCollectionManager()
        util = mgr.get_sensor_utilization()
        assert util["utilization"] == 0.0
        assert util["total_sensors"] == 0


class TestStats:
    def test_get_stats(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        mgr.create_task(SensorType.RADAR, "A", priority=Priority.HIGH)
        mgr.create_task(SensorType.EO_IR, "B", priority=Priority.LOW)
        plan = mgr.create_plan()
        stats = mgr.get_stats()
        assert stats["total_tasks"] == 2
        assert stats["total_sensors"] == 1
        assert stats["total_plans"] == 1
        assert stats["by_task_status"]["pending"] == 2
        assert stats["by_priority"]["HIGH"] == 1
        assert stats["by_priority"]["LOW"] == 1

    def test_reset(self):
        mgr = ISRCollectionManager()
        mgr.register_sensor(Sensor(sensor_id="S1", sensor_type=SensorType.RADAR))
        mgr.create_task(SensorType.RADAR, "A")
        mgr.create_plan()
        mgr.reset()
        assert mgr.get_all_sensors() == []
        assert mgr.get_all_tasks() == []
        assert mgr.get_stats()["total_plans"] == 0
