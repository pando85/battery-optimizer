"""Resident shadow service (ADR-0004, plan §16 Phase C).

Read-only Home Assistant and Prometheus adapters, the offline MPC kernel run live
in a bounded worker, time-bounded shadow intents that are recorded and never sent,
and the shadow-vs-native comparison.  See ``docs/service.md``.
"""
