import json
import time
from bootstrap.boot_strap import run_data_collection

TASK_BATTERY = "TEST"
TRACKING_CONFIG = "debug_video_data"
num_trials = 1


# Still need to collect neural mppi data
# controllers = ["neural_mppi"] # ["_pid", "analytical_mppi", "neural_mppi"]

controllers = ["_pid", "analytical_mppi", "neural_mppi"]
physics_models = ["dyn", "pyb", "pyb_drag"]

for controller in controllers:
	for physics_model in physics_models:
		##### Update the tracking config for this group of trials
		with open("./configs/tracking/tracking_config.json", "r") as tracking_f:
			tracking_config = json.load(tracking_f)
			phys, ctrl = tuple(controller.split("_"))
			tracking_config["CONTROLLER"] = ctrl
			tracking_config["PHYSICS"] = physics_model
		with open("./configs/tracking/tracking_config.json", "w") as tracking_f:
			json.dump(tracking_config, tracking_f, indent="\t", sort_keys=True)
		##### Update the mppi config if necessary
		if "mppi" in controller:
			with open("./configs/mppi_config.json", "r") as mppi_f:
				mppi_config = json.load(mppi_f)
				if phys == "analytical":
					mppi_config["DYNAMICS_MODEL"] = "AnalyticalModel"
				else:
					mppi_config["DYNAMICS_MODEL"] = "SampleLearnedModel"
			with open("./configs/mppi_config.json", "w") as mppi_f:
				json.dump(mppi_config, mppi_f, indent="\t", sort_keys=True)
		

		for trial in range(num_trials):
			run_data_collection(TASK_BATTERY, TRACKING_CONFIG)
			time.sleep(5)
			