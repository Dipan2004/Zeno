"""
ZENO Autonomy Controller - Controlled Multi-Step Execution

Phase 6: Controlled Autonomy

Purpose:
Allow ZENO to continue multi-step tasks automatically after initial planning,
without requiring the user to re-trigger each step.

RULES (HARD - DO NOT CHANGE):
- NO background threads
- NO autonomous triggers without user input
- MAX_STEPS = 5
- STOP immediately if any task fails
- Only active during a single user command session
- Does NOT modify Orchestrator logic
- Wraps execution ONLY

Architecture position:
    User Input → FastRouter → PlannerAgent → AutonomyController → Orchestrator → Agents

This module is called AFTER planner.plan() returns a task_graph.
It executes and optionally continues with follow-up tasks.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Hard cap on autonomous continuation steps.
# Step 1 = initial plan execution (always runs).
# Steps 2-5 = follow-up plans if planner says work is incomplete.
MAX_STEPS = 5


class AutonomyController:
    """
    Wraps Orchestrator execution with controlled multi-step continuation.

    Does NOT modify Orchestrator, PlannerAgent, or any Agent.
    Reads results → asks PlannerAgent if complete → continues if not.

    Usage:
        autonomy = AutonomyController(orchestrator, planner)
        results, explanation = autonomy.execute(task_graph, context_snapshot, original_input)
    """

    def __init__(self, orchestrator, planner):
        """
        Args:
            orchestrator: Orchestrator instance (registered with all agents)
            planner:      PlannerAgent instance (used for is_complete + plan_followup)
        """
        self.orchestrator = orchestrator
        self.planner      = planner
        logger.info("AutonomyController initialized (MAX_STEPS=%d)", MAX_STEPS)

    def execute(
        self,
        initial_task_graph,
        context_snapshot,
        original_input: str,
    ) -> tuple:
        """
        Execute a task graph, then optionally continue with follow-up tasks.

        Args:
            initial_task_graph: TaskGraph returned by planner.plan()
            context_snapshot:   ContextSnapshot used for planning
            original_input:     Original user command (for follow-up planning)

        Returns:
            (results: Dict[task_id → TaskResult], final_explanation: str)
            - results:           Merged dict of all task results across all steps
            - final_explanation: Explanation from the last planning step
        """
        tasks = list(initial_task_graph.tasks.values())
        all_results: Dict[str, Any] = {}
        step = 0

        while step < MAX_STEPS:
            step_label = f"step {step + 1}/{MAX_STEPS}"
            logger.info(
                "AutonomyController %s — executing %d task(s)",
                step_label, len(tasks)
            )

            # Execute current set of tasks via Orchestrator (unchanged)
            results = self.orchestrator.execute_plan(tasks)
            all_results.update(results)

            # STOP if any task failed — never continue on failure
            failed = [tid for tid, r in results.items() if not r.success]
            if failed:
                logger.info(
                    "AutonomyController stopping after %s — %d task(s) failed: %s",
                    step_label, len(failed), failed
                )
                break

            # Ask planner if the job is complete
            if self.planner.is_complete(results):
                logger.info(
                    "AutonomyController: planner reports complete after %s",
                    step_label
                )
                break

            # Not complete — get follow-up tasks (costs one more LLM call)
            step += 1
            if step >= MAX_STEPS:
                logger.warning(
                    "AutonomyController reached MAX_STEPS=%d — stopping to avoid runaway execution",
                    MAX_STEPS
                )
                break

            logger.info(
                "AutonomyController: planning follow-up (%s)...", step_label
            )
            try:
                follow_up_graph, follow_up_explanation = self.planner.plan_followup(
                    results=results,
                    context_snapshot=context_snapshot,
                    original_input=original_input,
                )
            except Exception as e:
                logger.error(
                    "AutonomyController: follow-up planning failed: %s", e,
                    exc_info=True
                )
                break

            if not follow_up_graph or not follow_up_graph.tasks:
                logger.info("AutonomyController: follow-up plan is empty — stopping")
                break

            tasks = list(follow_up_graph.tasks.values())

        logger.info(
            "AutonomyController finished after %d step(s), %d total task(s) executed",
            step, len(all_results)
        )
        return all_results