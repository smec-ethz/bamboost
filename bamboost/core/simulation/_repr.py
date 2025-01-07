import pkgutil
from typing import TYPE_CHECKING, Sized

if TYPE_CHECKING:
    from bamboost.core.simulation import Simulation


def simulation_html_repr(simulation: "Simulation") -> str:
    metadata = simulation.metadata
    parameters_filtered = {
        k: "..."
        if isinstance(v, Sized) and not isinstance(v, str) and len(v) > 5
        else v
        for k, v in simulation.parameters.items()
    }

    def get_pill_div(text: str, color: str) -> str:
        return (
            f'<div class="status" style="background-color:'
            f'var(--bb-{color});">{text}</div>'
        )

    status_options = {
        "finished": get_pill_div("finished", "green"),
        "failed": get_pill_div("failed", "red"),
        "initialized": get_pill_div("initialized", "grey"),
    }
    submitted_options = {
        True: get_pill_div("Submitted", "green"),
        False: get_pill_div("Not submitted", "grey"),
    }

    from jinja2 import Template

    html_string = pkgutil.get_data("bamboost", "_repr/simulation.html").decode()
    icon = pkgutil.get_data("bamboost", "_repr/icon.txt").decode()
    template = Template(html_string)

    return template.render(
        uid=simulation.name,
        icon=icon,
        tree=repr(simulation.files).replace("\n", "</br>").replace(" ", "&nbsp;"),
        parameters=parameters_filtered,
        note=metadata.get("description"),
        status=status_options.get(
            metadata.get("status", "Initiated"),
            f'<div class="status">{metadata["status"]}</div>',
        ),
        submitted=submitted_options[metadata.get("submitted", False)],
        timestamp=metadata.get("created_at", "N/A"),
    )
