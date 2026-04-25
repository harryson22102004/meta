"""
OpenEnv-compliant server for PROMETHEUS.

Run:
    uvicorn prometheus_server:app --host 0.0.0.0 --port 8000
"""

from openenv.core import create_fastapi_app

from prometheus.openenv_wrapper import (
    InvestigationAction,
    InvestigationObservation,
    PrometheusOpenEnv,
)

app = create_fastapi_app(
    env=PrometheusOpenEnv,
    action_cls=InvestigationAction,
    observation_cls=InvestigationObservation,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
