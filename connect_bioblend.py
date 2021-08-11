import os
import random
import bioblend
from bioblend.galaxy import GalaxyInstance
from bioblend.galaxy import histories
from time import sleep

server='http://127.0.0.1:9090/'
key = 'd14e88e66c9e56d63a22b75423866177'
file_name = "sample_tf_code.py"


gi = GalaxyInstance(server, key=key)
history = histories.HistoryClient(gi)

rnd_int = random.randint(1, 10000000)

print(rnd_int)

new_history = history.create_history(str(rnd_int))
print(new_history)
uploaded_dataset = gi.tools.upload_file(file_name, new_history["id"])

sleep(20)

tool_name = "run_jupyter_job"
hist_id = new_history["id"]
file_path = uploaded_dataset["outputs"][0]["id"]
tool_inputs = {"inputs": {"select_file": file_path}}
tool_run = gi.tools.run_tool(hist_id, tool_name, tool_inputs)
tool_run["jobs"][0]["state"]
