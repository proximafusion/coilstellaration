import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
logging.getLogger("src").setLevel(logging.DEBUG)

coilbench_role = os.environ.get("COILBENCH_ROLE", "leaderboard").lower()

match coilbench_role:
    case "leaderboard":
        from app_leaderboard import app_leaderboard

        app_leaderboard.launch()
    case "runner":
        from app_runner import app_runner

        app_runner.launch()
