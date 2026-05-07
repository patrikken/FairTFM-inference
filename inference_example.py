"""
FairTFM Inference Example - Fairness Analysis on ACS PUMS Income Prediction

This example demonstrates how to:
1. Load ACS PUMS dataset
2. Load and fit FairTFM classifiers and a baseline XGBoost model
3. Analyze model fairness using accuracy and fairness performance
"""


import numpy as np
from fairtfm import FairTFMClassifier, ACSPumsDataset, get_default_device, set_randomness_seed, compute_fairness_metrics
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier  

def fairness_analysis_example():
    """Main example: Fairness analysis with FairTFM and accuracy and fairness performance."""
    
    set_randomness_seed(42)
    device = get_default_device()
    print(f"Device: {device}\n")
    
    # ============================================================================
    # 1. LOAD DATA
    # ============================================================================
    print("=" * 70)
    print("1. LOADING ACS PUMS INCOME DATASET")
    print("=" * 70)
    
    dataset = ACSPumsDataset(
        acs_task="acs_income",
        states=["AL"],
        sensitive_attr_name="SEX",
        max_samples=10000,
        test_size=.2,
        seed_everything=42
    )
    dataset.preprocess()
    X_train, X_test, y_train, y_test, s_train, s_test = dataset.get_splits()
    
    print(f"Dataset: {dataset.name}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Features: {X_train.shape[1]} | Classes: {len(np.unique(y_train))}")
    print(f"Sensitive attribute (SEX): groups {np.unique(s_train)}\n")
    
    # ============================================================================
    # 2. LOAD AND FIT CLASSIFIER
    # ============================================================================
    print("=" * 70)
    print("2. FITTING CLASSIFIERS")
    print("=" * 70)


    models = {
        "FairTFM-0.7": FairTFMClassifier(model="checkpoints/FairTFM-0.7/checkpoint_epoch_10000.pt",device=device),
        "FairTFM-25": FairTFMClassifier(model="checkpoints/FairTFM-25/checkpoint_epoch_10000.pt",device=device),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric='logloss'),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}

    for model_name, clf in models.items(): 
        # record training time for each model  
        if isinstance(clf, FairTFMClassifier): 
            clf.fit(X_train, y_train, s_train)
            y_prod = clf.predict_proba(X_test, s_test)
        else:
            clf.fit(X_train, y_train)
            y_prod = clf.predict_proba(X_test) 
        end_time = time.time() 
        # ============================================================================
        # 3. FAIRNESS ANALYSIS (MAIN FOCUS)
        # ============================================================================
        print("=" * 70)
        print(f" Fitting {model_name}")
        print("=" * 70) 
        results[model_name] = compute_fairness_metrics(y_prob=y_prod, y_test=y_test, s_test=s_test) 
        
    
    print("\n📊 PERFORMANCE METRICS:")

    for model_name in results.keys():
        metrics = results[model_name]
        print(f"\n{model_name}:")
        print(f"  ROC AUC:                 {metrics['auc']:.4f}")  
        print(f"  Accuracy:                {metrics['accuracy']:.4f}")  
        print(f"  Demographic Parity Diff: {metrics['demographic_parity_difference']:.4f}")
        print(f"  Equalized Odds Diff:     {metrics['equalized_odds_difference']:.4f}")
        print(f"  Equal Opportunity Diff:  {metrics['equal_opportunity_difference']:.4f}")
        print(f"  Fitting and Inference Time (s):       {metrics['training_time_seconds']:.4f}")
 




if __name__ == "__main__":
    fairness_analysis_example()
    
    # custom_data_example() 
    
    print("\n✅ Examples completed!")
