from __future__ import annotations

import unittest
from unittest import mock

import dashboard.app as app


class ControllerStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        app.remediation_thread_started = False
        app.scaler_thread_started = False
        app.release_controller_lock()

    def test_controllers_start_when_manager_and_lock_available(self) -> None:
        with (
            mock.patch("dashboard.app.is_swarm_manager", return_value=True),
            mock.patch("dashboard.app.acquire_controller_lock", return_value=True),
            mock.patch("dashboard.app.ensure_remediation_thread") as remediation,
            mock.patch("dashboard.app.ensure_scaler_thread") as scaler,
        ):
            result = app.start_controllers_if_eligible()

        self.assertTrue(result)
        remediation.assert_called_once()
        scaler.assert_called_once()

    def test_controllers_do_not_start_without_manager(self) -> None:
        with (
            mock.patch("dashboard.app.is_swarm_manager", return_value=False),
            mock.patch("dashboard.app.acquire_controller_lock") as acquire,
            mock.patch("dashboard.app.ensure_remediation_thread") as remediation,
            mock.patch("dashboard.app.ensure_scaler_thread") as scaler,
        ):
            result = app.start_controllers_if_eligible()

        self.assertFalse(result)
        acquire.assert_not_called()
        remediation.assert_not_called()
        scaler.assert_not_called()

    def test_controllers_do_not_start_without_lock(self) -> None:
        with (
            mock.patch("dashboard.app.is_swarm_manager", return_value=True),
            mock.patch("dashboard.app.acquire_controller_lock", return_value=False),
            mock.patch("dashboard.app.ensure_remediation_thread") as remediation,
            mock.patch("dashboard.app.ensure_scaler_thread") as scaler,
        ):
            result = app.start_controllers_if_eligible()

        self.assertFalse(result)
        remediation.assert_not_called()
        scaler.assert_not_called()

    def test_controller_threads_only_started_once(self) -> None:
        started_threads = 0

        def dummy_thread(target, daemon):  # type: ignore[override]
            nonlocal started_threads
            started_threads += 1

            class _Thread:
                def start(self) -> None:
                    return None

            return _Thread()

        with (
            mock.patch("dashboard.app.is_swarm_manager", return_value=True),
            mock.patch("dashboard.app.acquire_controller_lock", return_value=True),
            mock.patch("dashboard.app.threading.Thread", side_effect=dummy_thread),
        ):
            app.start_controllers_if_eligible()
            app.start_controllers_if_eligible()

        self.assertEqual(started_threads, 2)


if __name__ == "__main__":
    unittest.main()
