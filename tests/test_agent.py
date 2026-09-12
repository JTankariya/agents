import pandas as pd

from packages.agent_core.agent import DataAnalystAgent
from packages.agent_core.contracts import ModelPlan
from packages.agent_core.sandbox import SandboxPolicy


class RepairingGateway:
    def __init__(self) -> None:
        self.repairs: list[tuple[str, str, str, str]] = []

    def generate(self, user_request: str, dataset_context: str) -> ModelPlan:
        return ModelPlan(
            status="ok",
            message="first",
            code='result = df["does_not_exist"]',
        )

    def repair(
        self,
        user_request: str,
        dataset_context: str,
        previous_code: str,
        safe_error: str,
    ) -> ModelPlan:
        self.repairs.append((user_request, dataset_context, previous_code, safe_error))
        return ModelPlan(
            status="ok",
            message="repaired",
            code='result = df[["value"]].sum().to_frame(name="total")',
        )


class GuardrailGateway:
    def generate(self, user_request: str, dataset_context: str) -> ModelPlan:
        return ModelPlan(
            status="guardrail",
            message="GUARDRAIL_TRIGGERED: Data questions only.",
        )


def test_agent_repairs_once():
    gateway = RepairingGateway()
    agent = DataAnalystAgent(gateway, SandboxPolicy(timeout_seconds=5))
    result = agent.run(
        user_request="sum value",
        dataframe=pd.DataFrame({"value": [1, 2, 3]}),
    )
    assert result.status == "success"
    assert result.attempts == 2
    assert len(gateway.repairs) == 1
    assert result.output is not None
    assert result.output.result.iloc[0, 0] == 6
    assert "does_not_exist" in gateway.repairs[0][2]


def test_agent_returns_guardrail():
    agent = DataAnalystAgent(GuardrailGateway(), SandboxPolicy())
    result = agent.run(
        user_request="write a poem",
        dataframe=pd.DataFrame({"value": [1]}),
    )
    assert result.status == "guardrail"
    assert result.attempts == 0
    assert "GUARDRAIL_TRIGGERED" in result.message
