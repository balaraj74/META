"""TRIAGE environment package — OpenEnv-compatible hospital simulation."""

from triage.env.hospital_env import HospitalEnv
from triage.env.openenv_adapter import TriageOpenEnv

__all__ = ["HospitalEnv", "TriageOpenEnv"]
