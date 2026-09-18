"""Regression tests for direct predecessor RCA-file reuse."""

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RUN_DOCTOR_SCRIPT = Path(__file__).resolve().parent.parent / "run-doctor.py"


def load_run_doctor():
    spec = importlib.util.spec_from_file_location("run_doctor", RUN_DOCTOR_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_DOCTOR = load_run_doctor()


def valid_rca_entry(evidence_path):
    return {
        "severity": 3,
        "stack_layer": "deploy phase",
        "step_name": "ipi-install",
        "error_signature": "install timed out",
        "root_cause": "cluster operators never went available",
        "raw_error": "timed out waiting for the condition",
        "infrastructure_failure": False,
        "job_url": "https://prow.example/job/123",
        "job_name": "periodic-ci-example",
        "release": "4.20",
        "remediation": "retry the install",
        "finished": "2026-08-13T00:00:00Z",
        "confidence": "high",
        "analysis_gaps": [],
        "scenarios": [],
        "causal_chain": [
            {
                "cause": "install timed out",
                "evidence": f"{evidence_path}:1",
                "quote": "timed out waiting for the condition",
            }
        ],
    }


class PredecessorReuseTests(unittest.TestCase):
    def test_stable_names_separate_release_and_pr_namespaces(self):
        release_name = RUN_DOCTOR._analysis_output_name(
            "periodic/ci example", "12345", release="4.20"
        )
        pr_name = RUN_DOCTOR._analysis_output_name(
            "periodic/ci example", "12345", pr_number="42"
        )

        self.assertEqual(
            release_name, "release-4.20-job-periodic-ci-example-12345.json"
        )
        self.assertEqual(pr_name, "prs-job-pr42-periodic-ci-example-12345.json")
        self.assertNotEqual(release_name, pr_name)

    def test_predecessor_workdir_accepts_cli_and_environment(self):
        base_args = ["run-doctor.py", "--releases", "main", "--workdir", "/tmp/current"]
        with mock.patch.dict(
            os.environ, {"CI_DOCTOR_PREDECESSOR_WORKDIR": "/tmp/environment"}
        ):
            with mock.patch.object(sys, "argv", base_args):
                self.assertEqual(
                    RUN_DOCTOR.parse_args().predecessor_workdir, "/tmp/environment"
                )
            with mock.patch.object(
                sys, "argv", [*base_args, "--predecessor-workdir", "/tmp/explicit"]
            ):
                self.assertEqual(
                    RUN_DOCTOR.parse_args().predecessor_workdir, "/tmp/explicit"
                )

    def test_reuses_valid_direct_predecessor_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predecessor = root / "predecessor"
            current = root / "current"
            predecessor_evidence = predecessor / "artifacts" / "12345" / "build-log.txt"
            current_evidence = current / "artifacts" / "12345" / "build-log.txt"
            predecessor_evidence.parent.mkdir(parents=True)
            current_evidence.parent.mkdir(parents=True)
            predecessor_evidence.write_text("timed out waiting for the condition\n")
            current_evidence.write_text("timed out waiting for the condition\n")
            output_name = RUN_DOCTOR._analysis_output_name(
                "periodic-ci-example", "12345", release="4.20"
            )
            predecessor_output = predecessor / "jobs" / output_name
            predecessor_output.parent.mkdir()
            predecessor_output.write_text(
                json.dumps([valid_rca_entry(predecessor_evidence)])
            )
            output_path = current / "jobs" / output_name
            output_path.parent.mkdir()

            _, reused = RUN_DOCTOR._reuse_predecessor_analysis(
                output_path, predecessor, current
            )

            self.assertTrue(reused)
            copied = json.loads(output_path.read_text())
            self.assertEqual(
                copied[0]["causal_chain"][0]["evidence"], f"{current_evidence}:1"
            )

    def test_malformed_predecessor_file_is_a_reuse_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predecessor = root / "predecessor"
            current = root / "current"
            output_name = RUN_DOCTOR._analysis_output_name(
                "periodic-ci-example", "12345", release="4.20"
            )
            predecessor_output = predecessor / "jobs" / output_name
            predecessor_output.parent.mkdir(parents=True)
            predecessor_output.write_text("not json")
            output_path = current / "jobs" / output_name
            output_path.parent.mkdir(parents=True)

            _, reused = RUN_DOCTOR._reuse_predecessor_analysis(
                output_path, predecessor, current
            )

            self.assertFalse(reused)
            self.assertFalse(output_path.exists())

    def test_validation_failing_predecessor_file_is_a_reuse_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predecessor = root / "predecessor"
            current = root / "current"
            output_name = RUN_DOCTOR._analysis_output_name(
                "periodic-ci-example", "12345", release="4.20"
            )
            predecessor_output = predecessor / "jobs" / output_name
            predecessor_output.parent.mkdir(parents=True)
            predecessor_output.write_text("[]")
            output_path = current / "jobs" / output_name
            output_path.parent.mkdir(parents=True)

            _, reused = RUN_DOCTOR._reuse_predecessor_analysis(
                output_path, predecessor, current
            )

            self.assertFalse(reused)
            self.assertFalse(output_path.exists())

    def test_unsafe_predecessor_evidence_is_a_reuse_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predecessor = root / "predecessor"
            current = root / "current"
            outside_evidence = root / "outside.log"
            outside_evidence.write_text("timed out waiting for the condition\n")
            output_name = RUN_DOCTOR._analysis_output_name(
                "periodic-ci-example", "12345", release="4.20"
            )
            predecessor_output = predecessor / "jobs" / output_name
            predecessor_output.parent.mkdir(parents=True)
            predecessor_output.write_text(
                json.dumps([valid_rca_entry(outside_evidence)])
            )
            output_path = current / "jobs" / output_name
            output_path.parent.mkdir(parents=True)

            _, reused = RUN_DOCTOR._reuse_predecessor_analysis(
                output_path, predecessor, current
            )

            self.assertFalse(reused)
            self.assertFalse(output_path.exists())

    def test_reuse_miss_runs_fresh_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predecessor = root / "predecessor"
            current = root / "current"
            evidence = current / "artifacts" / "12345" / "build-log.txt"
            evidence.parent.mkdir(parents=True)
            evidence.write_text("timed out waiting for the condition\n")
            output_name = RUN_DOCTOR._analysis_output_name(
                "periodic-ci-example", "12345", release="4.20"
            )
            output_path = current / "jobs" / output_name
            output_path.parent.mkdir(parents=True)

            with mock.patch.object(
                RUN_DOCTOR,
                "_run_claude_session",
                return_value=(True, json.dumps([valid_rca_entry(evidence)])),
            ) as session:
                saved, result_path, errors, _ = RUN_DOCTOR._analyze_single_job(
                    {
                        "output_name": output_name,
                        "log_name": "test.log",
                        "artifacts_dir": "",
                        "job_url": "https://prow.example/job/123",
                        "job_name": "periodic-ci-example",
                    },
                    plugin_dir="",
                    model="",
                    agent_system_prompt="",
                    logs_dir=current / "logs",
                    workdir=current,
                    predecessor_workdir=predecessor,
                )

            session.assert_called_once()
            self.assertTrue(saved)
            self.assertEqual(result_path, str(output_path))
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
