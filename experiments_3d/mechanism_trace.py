"""Expose actual RDG actions and M5 corrections for local mechanism evidence."""
import dataclasses
from Agent.hybrid.rdg_nlns.agent import RDGNLNSAgent
from Agent.hybrid.rdg_nlns.repair import RepairMode


class MechanismTraceMixin:
    def _record_repair_plan(self,plan):
        self._last_repair_plan=plan

    def _select_learned_action(self,graph,excluded_actions=frozenset()):
        selection=super()._select_learned_action(graph,excluded_actions)
        self._last_selection=dataclasses.asdict(selection)
        return selection

    def _sample_training_action(self,graph):
        selection=super()._sample_training_action(graph)
        self._last_selection=dataclasses.asdict(selection)
        return selection

    def _attempt(self,snapshot,current,destroy_id,repair_id,*,deadline=None):
        self._last_repair_plan=None
        action,outcome,elapsed=super()._attempt(snapshot,current,destroy_id,repair_id,deadline=deadline)
        callback=getattr(self,'trace_callback',None)
        if callback is not None:
            mode=RepairMode(self._config.repair_modes[repair_id])
            proposal=self._last_repair_plan
            candidate=outcome.candidate if outcome else None
            callback({'epoch_s':snapshot.rolling_epoch,'destroy_action':destroy_id,'repair_action':repair_id,
                'destroy_operator':action.operator.name,'destroy_ratio':action.ratio,'repair_mode':mode.name,
                'policy_selection':getattr(self,'_last_selection',None),'ablation':dataclasses.asdict(self._config.ablation),
                'before_objective':current.objective_value,'candidate_objective':candidate.objective_value if candidate else None,
                'removed_column_ids':list(action.removed_column_ids),'before_projection_ids':list(proposal.selected_column_ids) if proposal is not None else [],
                'after_projection_ids':list(candidate.selected_column_ids) if candidate else [],
                'projection_valid':bool(outcome and outcome.validation.is_valid and candidate and candidate.is_feasible),
                'projection_corrections':list(outcome.corrections) if outcome else [],
                'issue_codes':list(outcome.validation.issue_codes) if outcome else ['empty_destroy_action'],
                'repair_seconds':elapsed})
        return action,outcome,elapsed


class TracedRDGAgent(MechanismTraceMixin,RDGNLNSAgent):
    pass
