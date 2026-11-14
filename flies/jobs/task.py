"""Core registry entry for the Workday-inspired simulation (legacy in legacy_task.py)."""
from flies.jobs.workday import SimulatedWorkdayEnvironment


class JobsFly(SimulatedWorkdayEnvironment):
    # The 3D world (server.serve) runs the best held-out checkpoint (see DEMO.md) on the level it was trained on,
    # with never-seen test-split forms.
    world_checkpoint = "workday_v3_p2/checkpoints/update_000020/brain.pt"
    world_args = {"level": 1, "split": "test"}

    def view(self):
        """For the world's station: what each of the 8 slots does right now (labels only, no profile values)."""
        acts = [(a.kind + " " + a.label) if a.kind == "enter" else a.label for a in self.available_actions()]
        return {"actions": (acts + [""] * 8)[:8], "stage": getattr(self, "stage", "")}
