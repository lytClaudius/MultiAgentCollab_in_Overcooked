from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action


@dataclass
class AgentDecision:
    action: Any
    ingredient_hint: Any = None
    info: Dict[str, Any] = field(default_factory=dict)


class AlgorithmicAgent(Agent):
    """Base agent that returns a low-level action plus an optional meta payload."""

    def __init__(self, name: str = "AlgorithmicAgent"):
        self.name = name
        self.teammate = None
        self.last_context: Dict[str, Any] = {}
        self.last_decision: Optional[AgentDecision] = None
        self.training_mode = False

    def reset(self, teammate=None):
        self.teammate = teammate
        self.last_context = {}
        self.last_decision = None

    def set_training_mode(self, training: bool):
        self.training_mode = training

    def build_context(self, state) -> Dict[str, Any]:
        return {
            "state": state,
            "mdp": getattr(self, "mdp", None),
            "agent_index": getattr(self, "agent_index", None),
            "teammate": self.teammate,
        }

    @staticmethod
    def _normalize_decision(decision) -> AgentDecision:
        if isinstance(decision, AgentDecision):
            return decision
        if isinstance(decision, tuple):
            if len(decision) == 2:
                return AgentDecision(decision[0], decision[1], {})
            if len(decision) == 3:
                return AgentDecision(decision[0], decision[1], decision[2] or {})
        return AgentDecision(decision, None, {})

    def decide(self, state, context: Dict[str, Any]) -> AgentDecision:
        raise NotImplementedError

    def action(self, state):
        context = self.build_context(state)
        decision = self._normalize_decision(self.decide(state, context))
        self.last_context = context
        self.last_decision = decision
        return decision.action, decision.ingredient_hint


class GreedyBaselineAgent(AlgorithmicAgent):
    """A lightweight greedy baseline with pluggable task and scoring hooks."""

    def __init__(
        self,
        task_selector: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
        action_scorer: Optional[Callable[[Any, Any, Any, Dict[str, Any]], float]] = None,
        candidate_actions: Optional[Callable[[Any, Dict[str, Any]], Sequence[Any]]] = None,
        ingredient_selector: Optional[Callable[[Any, Any, Any, Dict[str, Any]], Any]] = None,
        name: str = "GreedyBaselineAgent",
    ):
        super().__init__(name=name)
        self.task_selector = task_selector or self._default_task_selector
        self.action_scorer = action_scorer or self._default_action_scorer
        self.candidate_actions = candidate_actions or self._default_candidate_actions
        self.ingredient_selector = ingredient_selector or self._default_ingredient_selector

    def _default_task_selector(self, state, context):
        order_list = getattr(state, "current_k_order", None) or getattr(state, "order_list", None)
        if order_list:
            return order_list[0]
        return None

    def _default_candidate_actions(self, state, context):
        return list(Action.ALL_ACTIONS)

    def _default_action_scorer(self, state, action, task, context):
        if action == Action.INTERACT:
            return 2.0
        if action == Action.STAY:
            return -1.0
        return 0.0

    def _default_ingredient_selector(self, state, action, task, context):
        return task

    def decide(self, state, context):
        task = self.task_selector(state, context)
        candidates = list(self.candidate_actions(state, context))
        if not candidates:
            return AgentDecision(Action.STAY, None, {"task": task, "policy": "greedy"})

        best_action = max(candidates, key=lambda action: self.action_scorer(state, action, task, context))
        ingredient_hint = self.ingredient_selector(state, best_action, task, context)
        return AgentDecision(best_action, ingredient_hint, {"task": task, "policy": "greedy"})


class OpponentAwareSearchAgent(AlgorithmicAgent):
    """A policy shell for belief-conditioned minimax / adversarial search agents."""

    def __init__(
        self,
        opponent_type_estimator: Optional[Callable[[Any, Dict[str, Any]], Dict[str, float]]] = None,
        action_scorer: Optional[Callable[[Any, Any, Dict[str, float], Dict[str, Any]], float]] = None,
        candidate_actions: Optional[Callable[[Any, Dict[str, Any]], Sequence[Any]]] = None,
        name: str = "OpponentAwareSearchAgent",
    ):
        super().__init__(name=name)
        self.opponent_type_estimator = opponent_type_estimator or self._default_opponent_type_estimator
        self.action_scorer = action_scorer or self._default_action_scorer
        self.candidate_actions = candidate_actions or self._default_candidate_actions

    def _default_opponent_type_estimator(self, state, context):
        return {"cooperative": 0.5, "greedy": 0.5}

    def _default_candidate_actions(self, state, context):
        return list(Action.ALL_ACTIONS)

    def _default_action_scorer(self, state, action, opponent_belief, context):
        cooperation_prob = opponent_belief.get("cooperative", 0.5)
        if action == Action.INTERACT:
            return 1.0 + cooperation_prob
        if action == Action.STAY:
            return -0.25
        return 0.0

    def decide(self, state, context):
        opponent_belief = self.opponent_type_estimator(state, context)
        candidates = list(self.candidate_actions(state, context))
        if not candidates:
            return AgentDecision(Action.STAY, None, {"opponent_belief": opponent_belief, "policy": "opponent_aware"})

        best_action = max(candidates, key=lambda action: self.action_scorer(state, action, opponent_belief, context))
        return AgentDecision(best_action, None, {"opponent_belief": opponent_belief, "policy": "opponent_aware"})


class MultiAgentRLAgent(AlgorithmicAgent):
    """A thin RL-friendly wrapper for online policy inference and learning hooks."""

    def __init__(
        self,
        policy: Optional[Callable[[Any, Dict[str, Any]], Any]] = None,
        update_fn: Optional[Callable[[Sequence[Any]], None]] = None,
        name: str = "MultiAgentRLAgent",
    ):
        super().__init__(name=name)
        self.policy = policy or self._default_policy
        self.update_fn = update_fn
        self.trajectory = []

    def _default_policy(self, state, context):
        return AgentDecision(Action.STAY, None, {"policy": "rl_stub"})

    def decide(self, state, context):
        decision = self._normalize_decision(self.policy(state, context))
        return decision

    def observe_transition(self, transition):
        self.trajectory.append(transition)

    def learn(self):
        if self.update_fn is not None and self.trajectory:
            self.update_fn(self.trajectory)
        self.trajectory = []