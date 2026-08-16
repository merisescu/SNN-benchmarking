import optuna
import os

user = os.environ['POSTGRES_USER']
pwd = os.environ['POSTGRES_PASSWORD']
lif_db = os.environ['POSTGRES_DB']

if __name__ == "__main__":
    study = optuna.create_study(
        directions=['minimize','maximize'],
        sampler=optuna.samplers.TPESampler(constant_liar=True),
        study_name="NMNIST_LIF",
        storage=f"postgresql://{user}:{pwd}@pynest-sql:5432/{lif_db}",
        load_if_exists=True,
    )