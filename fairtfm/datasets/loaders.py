"""
ACS PUMS dataset loader for fairness tasks.
Used for testing inference on real fairness benchmark datasets.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from folktables import (
    ACSDataSource,
    ACSEmployment,
    ACSIncome,
    ACSMobility,
    ACSPublicCoverage,
    ACSTravelTime,
)


class ACSPumsDataset:
    """
    Loads American Community Survey (ACS) Public Use Microdata Sample (PUMS) tasks.
    Provides fairness benchmarks with configurable sensitive attributes.
    
    Example:
        >>> dataset = ACSPumsDataset(
        ...     acs_task="acs_income",
        ...     states=["CA"],
        ...     sensitive_attr_name="SEX"
        ... )
        >>> dataset.preprocess()
        >>> X_train, X_test, y_train, y_test, s_train, s_test = dataset.get_splits()
    """
    
    def __init__(
        self,
        acs_task: str = "acs_income",
        states: list = None,
        survey_year: int = 2018,
        sensitive_attr_name: str = "SEX",
        horizon: str = "1-Year",
        seed_everything: int = 42,
        max_samples: int = None,
        test_size: float = 0.2
    ):
        """
        Initialize ACS PUMS dataset loader.
        
        Args:
            acs_task: Task name ('acs_income', 'acs_employment', 'acs_mobility', 
                     'acs_public_coverage', 'acs_travel_time')
            states: List of state codes to include (e.g., ['CA', 'NY'])
            survey_year: Survey year (default 2018)
            sensitive_attr_name: Sensitive attribute ('SEX', 'RAC1P', 'AGEP')
            horizon: Survey horizon ('1-Year' or '5-Year')
            seed_everything: Random seed for reproducibility
            max_samples: Maximum samples to load (None = all)
            test_size: Proportion of data to use for testing
        """
        self.sensitive_attr_name = sensitive_attr_name
        self.acs_task = acs_task
        self.states = states if states is not None else ["AL"]
        self.survey_year = survey_year
        self.horizon = horizon
        self.seed = seed_everything
        self.max_samples = max_samples
        self.test_size = test_size
        
        # Validation
        if sensitive_attr_name not in ["SEX", "RAC1P", "AGEP"]:
            raise ValueError(f"Unsupported sensitive attribute: {sensitive_attr_name}")
        
        if acs_task not in [
            "acs_income", "acs_employment", "acs_mobility",
            "acs_public_coverage", "acs_travel_time",
        ]:
            raise ValueError(f"Unsupported ACS task: {acs_task}")
        
        state_name = "_".join(self.states) if self.states else "all_states"
        self.name = f"{acs_task}_{state_name}_{sensitive_attr_name}"
        
        self.data = None
        self.target = None
        self.sensitive_attr = None
        self.train_idx = None
        self.test_idx = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.s_train = None
        self.s_test = None

    def _get_task_class(self):
        """Get the appropriate task class for the ACS task."""
        task_map = {
            "acs_income": ACSIncome,
            "acs_employment": ACSEmployment,
            "acs_mobility": ACSMobility,
            "acs_public_coverage": ACSPublicCoverage,
            "acs_travel_time": ACSTravelTime,
        }
        return task_map[self.acs_task]

    def preprocess(self):
        """
        Download and preprocess the ACS PUMS dataset.
        Applies task-specific transformations and handles sensitive attributes.
        """
        # Download data
        data_source = ACSDataSource(
            survey_year=self.survey_year,
            horizon=self.horizon,
            survey="person",
            root_dir="data"
        )
        acs_data = data_source.get_data(
            states=self.states,
            download=True,
            random_seed=self.seed
        )
        
        # Subsample if needed
        nb_sample = min(len(acs_data), self.max_samples) if self.max_samples else len(acs_data)
        if self.max_samples is not None or len(self.states) == 0:
            acs_data = acs_data.sample(n=nb_sample, random_state=self.seed)
        
        # Apply task-specific transformations
        task_class = self._get_task_class()
        acs_data, self.target, _ = task_class.df_to_pandas(acs_data)
        
        # Extract sensitive attributes
        self.sensitive_attr = acs_data[self.sensitive_attr_name]
        
        # Keep only relevant features
        self.data = acs_data.drop(columns=[self.sensitive_attr_name])
        
        # Handle race attribute
        if self.sensitive_attr_name == "RAC1P":
            idx = acs_data[self.sensitive_attr_name].isin([1, 2])  # White and Black only
            self.data = self.data[idx]
            self.target = self.target[idx]
            self.sensitive_attr = self.sensitive_attr[idx]
        
        # binary age attribute
        elif self.sensitive_attr_name == "AGEP":
            self.sensitive_attr = self.sensitive_attr.apply(
                lambda x: 1 if x > 25 else 0
            )
        
        # Create train/test split
        self.train_idx, self.test_idx = train_test_split(
            range(len(self.data)),
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.target,
        )
        
        self.data = pd.get_dummies(self.data, drop_first=True).values.astype(np.float32) 
        y_sc = LabelEncoder()
        self.target = y_sc.fit_transform(self.target.squeeze())

        s_sc = LabelEncoder()
        self.sensitive_attr = s_sc.fit_transform(self.sensitive_attr.squeeze())

        self.train_idx, self.test_idx = train_test_split(
            range(len(self.data)),
            test_size=self.test_size,
            random_state=self.seed,
            stratify=self.target,
        ) 
        self.input_dim = self.data.shape[1]
        return self

    def get_splits(self):
        """
        Get train/test splits with features, labels, and sensitive attributes.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, s_train, s_test)
        """
        if self.data is None or self.target is None or self.sensitive_attr is None:
            raise ValueError("Dataset not preprocessed. Call preprocess() first.")
        scaler = StandardScaler()
        """ self.data = pd.DataFrame(
            scaler.fit_transform(self.data), columns=self.data.columns
        ) """

        scaler = StandardScaler() 

        self.X_train = self.data[self.train_idx]
        self.X_test = self.data[self.test_idx]
        self.y_train = self.target[self.train_idx]
        self.y_test = self.target[self.test_idx]
        self.s_train = self.sensitive_attr[self.train_idx]
        self.s_test = self.sensitive_attr[self.test_idx]

        self.X_train = scaler.fit_transform(self.X_train).astype("float32")
        self.X_test = scaler.transform(self.X_test).astype("float32")
        return (
            self.X_train, self.X_test,
            self.y_train, self.y_test,
            self.s_train, self.s_test
        )

    def __len__(self):
        """Return total number of samples."""
        if self.data is None:
            return 0
        return len(self.data)

    def __repr__(self):
        return f"ACSPumsDataset({self.name}, n_samples={len(self)})"
