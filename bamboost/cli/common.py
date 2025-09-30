from contextlib import contextmanager

import rich

console = rich.get_console()


@contextmanager
def task_status(start_msg: str, success_msg: str | None = None):
    """Context manager for a task with spinner + success/failure messages."""
    with console.status(start_msg, spinner="dots"):
        try:
            yield
        except Exception as e:
            console.print(f"[bold red]Task failed: {e}")
        else:
            if success_msg:
                console.print(f"[green]:heavy_check_mark: {success_msg}")
