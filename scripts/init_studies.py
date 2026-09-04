import optuna
import os

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
lif_db = os.environ['POSTGRES_DB']

if __name__ == "__main__":
    #optuna.delete_study(study_name="NMNIST_LIF_v2",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{lif_db}")
    #optuna.delete_study(study_name="MG_ALIF",storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{lif_db}")

    study = optuna.create_study(
        directions=['minimize','maximize'],
        sampler=optuna.samplers.TPESampler(constant_liar=True),
        study_name="NMNIST_LIF_sparse",
        storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{lif_db}",
        load_if_exists=True,
    )
    study = optuna.create_study(
            directions=['minimize','maximize'],
            sampler=optuna.samplers.TPESampler(constant_liar=True),
            study_name="NMNIST_ALIF_sparse",
            storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{lif_db}",
            load_if_exists=True,
        )

    study = optuna.create_study(
                directions=['minimize','maximize'],
                sampler=optuna.samplers.TPESampler(constant_liar=True),
                study_name="MG_LIF_dt",
                storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{lif_db}",
                load_if_exists=True,
            )

    study = optuna.create_study(
                directions=['minimize','maximize'],
                sampler=optuna.samplers.TPESampler(constant_liar=True),
                study_name="MG_ALIF_dt",
                storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{lif_db}",
                load_if_exists=True,
            )