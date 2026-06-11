import logging
import os

logging.basicConfig(level=logging.DEBUG)

coilbench_role = os.environ.get("COILBENCH_ROLE", "leaderboard").lower()

match coilbench_role:
    case "leaderboard":
        from app_leaderboard import demo as leaderboard_demo

        leaderboard_demo.launch()
    case "runner":
        from app_runner import demo as worker_demo

        worker_demo.launch()
