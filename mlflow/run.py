import mlflow 

parameters={
	
  "epoch": 25,
  "lr": 0.001

}

experiment_name = "mhistresnet_13"
entry_point = "main"

mlflow.projects.run(
 uri = ".",
 entry_point = entry_point,
 parameters = parameters,
 experiment_name = experiment_name


	)