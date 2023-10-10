import optuna as o
from optuna_dashboard import set_objective_names
from typing import List, Dict
import time
import mlflow
from concurrent.futures import ProcessPoolExecutor, wait
from joblib import parallel_backend
from .metrics import metrics


class Hyperparam_Optimizer:
    def __init__(
        self,
        name: str,
        sampler: str,
        seed: int,
        path: str,
        n_trials: int,
        timeout: int,
        enabled_hyperparameters,
        optimization_metric: List,
        n_jobs: int = 1,
        selective_optimization: bool = False,
        toggle_classical_quant: bool = False,
        resume_study: bool = False,
        pool_process=True,
        pruner_startup_trials=10,
        pruner_warmup_steps=5,
        pruner_interval_steps=1,
        pruner_min_trials=20,
    ):
        # storage = self.initialize_storage(host, port, path, password)

        pruner = o.pruners.MedianPruner(
            n_startup_trials=pruner_startup_trials,
            n_warmup_steps=pruner_warmup_steps,
            interval_steps=pruner_interval_steps,
            n_min_trials=pruner_min_trials,
        )
        # pruner = o.pruners.PercentilePruner(10.0, n_warmup_steps=2, n_startup_trials=20)

        if sampler == "TPESampler":
            sampler = o.samplers.TPESampler(
                seed=seed, multivariate=True, constant_liar=True
            )
        elif sampler == "RandomSampler":
            sampler = o.samplers.RandomSampler(seed=seed)
        else:
            raise ValueError(f"Sampler {sampler} is not supported")

        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.timeout = timeout
        self.toggle_classical_quant = toggle_classical_quant
        self.selective_optimization = selective_optimization
        self.pool_process = pool_process

        self.enabled_hyperparameters = enabled_hyperparameters

        self.studies = []
        self.n_jobs = n_jobs

        n_studies = self.n_jobs if self.pool_process else 1

        direction = (
            "minimize"
            if metrics[
                self.optimization_metric.replace("Train_", "").replace("Val_", "")
            ]["s"]
            > 0
            else "maximize"
        )

        for n_it in range(n_studies):
            resume_study = resume_study or (n_studies > 1 and n_it != 0)

            self.studies.append(
                o.create_study(
                    pruner=pruner,
                    sampler=sampler,
                    # directions=["maximize", "minimize", "maximize"],
                    direction=direction,
                    load_if_exists=resume_study,
                    study_name=name,
                    storage=f"sqlite:///{path}",
                )
            )

            self.studies[-1].set_metric_names([self.optimization_metric])

    def set_variable_parameters(self, model_parameters, instructor_parameters):
        assert isinstance(model_parameters, Dict)
        assert isinstance(instructor_parameters, Dict)

        def select_hyperparams(pair):
            key, _ = pair
            if len(self.enabled_hyperparameters) == 0:
                return True
            return key in self.enabled_hyperparameters

        self.variable_model_parameters = dict(
            filter(select_hyperparams, model_parameters.items())
        )
        self.variable_instructor_parameters = dict(
            filter(select_hyperparams, instructor_parameters.items())
        )

    def set_fixed_parameters(self, model_parameters, instructor_parameters):
        assert isinstance(model_parameters, Dict)
        assert isinstance(instructor_parameters, Dict)

        def select_hyperparams(pair):
            key, _ = pair
            if len(self.enabled_hyperparameters) == 0:
                return True
            return key not in self.enabled_hyperparameters

        self.fixed_model_parameters = dict(
            filter(select_hyperparams, model_parameters.items())
        )
        self.fixed_instructor_parameters = dict(
            filter(select_hyperparams, instructor_parameters.items())
        )

    @staticmethod
    def create_model():
        raise NotImplementedError("Create Model method must be set!")

    @staticmethod
    def create_instructor():
        raise NotImplementedError("Create Instructor method must be set!")

    @staticmethod
    def objective():
        raise NotImplementedError("Objective method must be set!")

    def early_stop_callback(self):
        # TODO: Implement early stopping
        return False

    def log_study(self):
        for study in self.studies:
            for trial in study.best_trials:
                mlflow.log_params(
                    {f"trial_{trial.number}_{k}": v for k, v in trial.params.items()}
                )

                mlflow.log_metric(f"trial_{trial.number}_accuracy", trial.values[0])
                mlflow.log_metric(f"trial_{trial.number}_loss", trial.values[1])

    def update_variable_parameters(self, trial, parameters, prefix=""):
        updated_variable_parameters = dict()
        choice = None  # indicate that this level is not a choice

        for parameter, value in parameters.items():
            # the following section removes any reserved strings from the parameter name and draws a choice if possible
            if "_range_quant" in parameter:
                if not self.toggle_classical_quant and self.selective_optimization:
                    continue  # continue if its a quantum parameter and we are classical
                param_name = parameter.replace("_range_quant", "")
            elif "_range" in parameter:
                if self.toggle_classical_quant and self.selective_optimization:
                    continue  # continue if its a classical parameter and we are quantum
                param_name = parameter.replace("_range", "")
            elif "_choice" in parameter:
                param_name = parameter.replace("_choice", "")
                assert type(value) == dict
                # if we have the choice; go and ask the trial what to do
                choice = trial.suggest_categorical(param_name, value.keys())
            else:
                # there is nothing fancy going on, copy the parameter name as is
                param_name = parameter

            # now, check if the hyperparameter is nested, i.e. there is another level below
            if isinstance(value, Dict):
                # this is a nested parameter, check if we had the choice
                if choice is not None:
                    # we want to skip through this choice (i.e. not iterate at this level)
                    # therefore create a new dict
                    updated_variable_parameters[param_name] = {}
                    # and assign the result as a new dict to the only key in this dict (this is just to preserve the structure given by the config)
                    # note that we preselect value[choice] and modify the prefix such that it includes the choice (to not get duplicates later when running trial.suggest(..))
                    updated_variable_parameters[param_name][
                        choice
                    ] = self.update_variable_parameters(
                        trial, value[choice], prefix=f"{prefix}{param_name}_{choice}_"
                    )
                else:
                    # the easy case; just go one level deeper and pass as prefix the current prefix (in case we are multilevel) as well as the current parameter name
                    updated_variable_parameters[
                        param_name
                    ] = self.update_variable_parameters(
                        trial, value, prefix=f"{prefix}{param_name}_"
                    )

                # as this is just a "virtual" level, there is no reason the check the following
                continue
            # ok, so this is not nested, and therefore can only be a list (i.e. a _range hyperparameter)
            elif not isinstance(value, List):
                raise RuntimeError(
                    "Provides parameter is not a dictionary or a list. Cannot infer hyperparameters."
                )

            # ----
            # here ends the recursive call from previous section

            # if we have three values (-> no bool) and they are not categorical (str) and the last one is a str (linear/log)
            if (
                len(value) == 3
                and not isinstance(value[0], str)
                and not isinstance(value[1], str)
                and isinstance(value[2], str)
            ):
                low = value[0]
                high = value[1]

                # if the third parameter specifies the scale
                if isinstance(value[2], str):
                    if value[2] == "log":
                        log = True
                        step = None
                    else:
                        log = False
                        step = value[0]
                else:
                    log = False
                    step = value[2]

                #
                if isinstance(low, float) and isinstance(high, float):
                    updated_variable_parameters[param_name] = trial.suggest_float(
                        prefix + param_name, value[0], value[1], step=step, log=log
                    )
                elif isinstance(low, int) and isinstance(high, int):
                    updated_variable_parameters[param_name] = trial.suggest_int(
                        prefix + param_name, value[0], value[1], step=1, log=log
                    )
                else:
                    raise ValueError(
                        f"Unexpected type of range for trial suggestion for parameter {param_name}. Expected one of 'float' or 'int', got [{type(low)}, {type(high)}]."
                    )

            else:
                updated_variable_parameters[param_name] = trial.suggest_categorical(
                    prefix + param_name, value
                )

        return updated_variable_parameters

    def minimize(self):
        self.studies[0].optimize(
            self.run_trial, n_trials=self.n_trials, n_jobs=self.n_jobs
        )

        return
        if self.pool_process:
            with ProcessPoolExecutor(max_workers=len(self.studies)) as pool:
                futures = []
                for study in self.studies:
                    futures.append(
                        pool.submit(
                            study.optimize, self.run_trial, n_trials=self.n_trials
                        )
                    )

                wait(futures)
        else:
            with parallel_backend("multiprocessing"):
                self.studies[0].optimize(
                    self.run_trial, n_trials=self.n_trials, n_jobs=self.n_jobs
                )

    def run_trial(self, trial):
        updated_variable_model_parameters = self.update_variable_parameters(
            trial, self.variable_model_parameters
        )
        model_parameters = (
            self.fixed_model_parameters | updated_variable_model_parameters
        )
        model = self.create_model(**model_parameters)

        # gather all parameters and create instructor
        updated_variable_instructor_parameters = self.update_variable_parameters(
            trial, self.variable_instructor_parameters
        )
        instructor_parameters = (
            self.fixed_instructor_parameters | updated_variable_instructor_parameters
        )
        instructor_parameters[
            "model"
        ] = model  # update this single parameter using the returned model
        instructor_parameters["report_callback"] = trial.report
        instructor_parameters["early_stop_callback"] = self.early_stop_callback
        instructor = self.create_instructor(**instructor_parameters)

        metric = self.objective(
            instructor=instructor,
            trial=trial,
        )

        return metric[self.optimization_metric]
        # return [metrics["accuracy"], metrics["loss"], metrics["perfect_lcag"]]
