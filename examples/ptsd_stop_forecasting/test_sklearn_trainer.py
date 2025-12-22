"""
Minimal test script for SklearnModule and SklearnTrainer.
Demonstrates usage with Ridge regression on PTSD forecasting data.
"""
import os
import sys
from pathlib import Path

# Add utils to path
# sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import add_to_path
add_to_path(__file__)

import pytorch_lightning as pl
from datasets import load_from_disk
import torch
import pickle

from src import (
    get_default_args, get_logger,
    get_datasetDict, MIDataLoaderModule,
    SklearnModule, SklearnTrainer,
    RidgeForecastModel, LassoForecastModel,
    AutoRegressiveRidge, AutoRegressiveLasso
)


def test_sklearn_trainer_minimal(args):
    """Minimal test with small dataset."""
    print("=" * 80)
    print("MINIMAL TEST: Ridge Regression with Sklearn Trainer")
    print("=" * 80)
    
    # Override with minimal settings for testing
    # args.train_batch_size = 8
    # args.val_batch_size = 16
    args.output_dir = '/tmp/sklearn_trainer_test'
    args.workspace = 'test'
    args.project_name = 'sklearn_trainer_test'
    args.experiment_name = 'ridge_minimal_test'
    
    # Set seed
    pl.seed_everything(args.seed if hasattr(args, 'seed') else 42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = load_from_disk(args.data_dir)
    
    # Create dataset splits
    datasetDict = get_datasetDict(train_data=data, val_folds=args.val_folds, test_folds=args.test_folds)
    for key in datasetDict:
        datasetDict[key] = datasetDict[key].with_format('torch')
    
    # Create dataloaders
    dataloaderModule = MIDataLoaderModule(args, datasetDict)
    
    print(f"Train dataset size: {len(dataloaderModule.train_dataloader())}")
    print(f"Val dataset size: {len(dataloaderModule.val_dataloader())}")
    if dataloaderModule.test_dataloader():
        print(f"Test dataset size: {len(dataloaderModule.test_dataloader())}")
    
    # Get logger (optional - can be None for testing)
    try:
        logger = get_logger('comet', 
                          workspace=args.workspace,
                          project_name=args.project_name,
                          experiment_name=args.experiment_name,
                          save_dir=args.output_dir)
        logger.log_hyperparams(args.__dict__)
        print("\nLogger initialized successfully")
    except Exception as e:
        print(f"\nWarning: Could not initialize logger: {e}")
        print("Continuing without logger...")
        logger = None
    
    # Create sklearn model (Ridge regression with feature selection)
    print("\n" + "=" * 80)
    print("1. TESTING WITH RIDGE REGRESSION")
    print("=" * 80)
    model = RidgeForecastModel(alpha=args.weight_decay, random_state=args.seed)
    
    # Create module
    module = SklearnModule(args, model)
    
    # Create trainer
    trainer = SklearnTrainer(output_dir=args.output_dir, logger=logger)
    
    # Initial validation (like Lightning)
    # print("\n--- Initial Validation ---")
    # trainer.validate(module, dataloaderModule.val_dataloader())
    
    # Train
    print("\n--- Training ---")
    results = trainer.fit(module, 
                         train_dataloader=dataloaderModule.train_dataloader(),
                         val_dataloader=dataloaderModule.val_dataloader())
    
    print(f"\nFinal training metrics: {results['train']}")
    print(f"Final validation metrics: {results['val']}")
    
    # Test (if test set available)
    if dataloaderModule.test_dataloader():
        print("\n--- Testing ---")
        test_results = trainer.test(module, dataloaderModule.test_dataloader())
        print(f"Test metrics: {test_results}")
    
    # Save predictions
    print("\n--- Saving Results ---")
    pred_file = os.path.join(args.output_dir, 'predictions.pkl')
    with open(pred_file, 'wb') as f:
        pickle.dump(module.predictions, f)
    print(f"Predictions saved to {pred_file}")
    
    metrics_file = os.path.join(args.output_dir, 'metrics.pkl')
    with open(metrics_file, 'wb') as f:
        pickle.dump(module.metrics, f)
    print(f"Metrics saved to {metrics_file}")
    
    if logger:
        logger.experiment.end()
    
    print("\n" + "=" * 80)
    print("MINIMAL TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


def test_hyperparameter_search():
    """Test hyperparameter search functionality."""
    print("\n\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH TEST")
    print("=" * 80)
    
    # Get default args
    args = get_default_args()
    
    # Override with minimal settings
    args.train_batch_size = 8
    args.val_batch_size = 16
    args.output_dir = '/tmp/sklearn_trainer_hparam_test'
    args.workspace = 'test'
    args.project_name = 'sklearn_trainer_test'
    args.experiment_name = 'ridge_hparam_search'
    
    # Set seed
    pl.seed_everything(args.seed if hasattr(args, 'seed') else 42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = load_from_disk(args.data_dir)
    
    # Create dataset splits
    datasetDict = get_datasetDict(train_data=data, val_folds=args.val_folds, test_folds=args.test_folds)
    for key in datasetDict:
        datasetDict[key] = datasetDict[key].with_format('torch')
    
    # Create dataloaders
    dataloaderModule = MIDataLoaderModule(args, datasetDict)
    
    # Create logger (optional)
    try:
        logger = get_logger('comet',
                          workspace=args.workspace,
                          project_name=args.project_name,
                          experiment_name=args.experiment_name,
                          save_dir=args.output_dir)
        logger.log_hyperparams(args.__dict__)
    except:
        logger = None
    
    # Create sklearn model (adapter)
    model = RidgeForecastModel(random_state=42)
    
    # Create module
    module = SklearnModule(args, model)
    
    # Create trainer
    trainer = SklearnTrainer(output_dir=args.output_dir, logger=logger)
    
    # Define hyperparameter grid
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    }
    
    # Run hyperparameter search
    print("\n--- Grid Search ---")
    best_module = trainer.hyperparameter_search(
        module,
        param_grid,
        dataloaderModule.train_dataloader(),
        dataloaderModule.val_dataloader(),
        search_type='grid',
        cv=3
    )
    
    print(f"\nBest validation metrics: {best_module.metrics['val']}")
    
    # Test best model
    if dataloaderModule.test_dataloader():
        print("\n--- Testing Best Model ---")
        test_results = trainer.test(best_module, dataloaderModule.test_dataloader())
        print(f"Test metrics with best model: {test_results}")
    
    if logger:
        logger.experiment.end()
    
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


def test_different_models():
    """Test with different sklearn models (Ridge vs Lasso)."""
    print("\n\n" + "=" * 80)
    print("TESTING MULTIPLE SKLEARN MODELS")
    print("=" * 80)
    
    # Get default args
    args = get_default_args()
    
    # Override with minimal settings
    args.train_batch_size = 8
    args.val_batch_size = 16
    args.output_dir = '/tmp/sklearn_trainer_models_test'
    
    # Set seed
    pl.seed_everything(args.seed if hasattr(args, 'seed') else 42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = load_from_disk(args.data_dir)
    
    # Create dataset splits
    datasetDict = get_datasetDict(train_data=data, val_folds=args.val_folds, test_folds=args.test_folds)
    for key in datasetDict:
        datasetDict[key] = datasetDict[key].with_format('torch')
    
    # Create dataloaders
    dataloaderModule = MIDataLoaderModule(args, datasetDict)
    
    # Test different models (adapters)
    models_to_test = [
        ('Ridge', RidgeForecastModel(alpha=1.0, random_state=42)),
        ('Lasso', LassoForecastModel(alpha=1.0, random_state=42, max_iter=1000))
    ]
    
    results_comparison = {}
    
    for model_name, model in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing {model_name}")
        print(f"{'=' * 80}")
        
        # Create module
        module = SklearnModule(args, model)
        
        # Create trainer (no logger for this test)
        trainer = SklearnTrainer(output_dir=os.path.join(args.output_dir, model_name.lower()))
        
        # Train
        results = trainer.fit(module,
                             train_dataloader=dataloaderModule.train_dataloader(),
                             val_dataloader=dataloaderModule.val_dataloader())
        
        results_comparison[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"  Train MSE: {results['train']['mse']:.4f}")
        print(f"  Val MSE: {results['val']['mse']:.4f}")
        print(f"  Train Pearson: {results['train']['pearsonr']:.4f}")
        print(f"  Val Pearson: {results['val']['pearsonr']:.4f}")
    
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    for model_name, results in results_comparison.items():
        print(f"\n{model_name}:")
        print(f"  Val MSE: {results['val']['mse']:.4f}")
        print(f"  Val Pearson: {results['val']['pearsonr']:.4f}")
    
    print("\n" + "=" * 80)
    print("MULTIPLE MODELS TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


def check_subset_presence_from_masks(dataloader):
    """
    Determine which OOTS/OOSS-based subsets actually have data, based only
    on masks. This mirrors the logical conditions used in both
    MILightningModule and SklearnModule.compute_exhaustive_metrics.
    """
    has = {
        "ws_wt": False,
        "valset": False,
        "ws_oots": False,
        "wt_ooss": False,
        "oots_ooss": False,
        "oots": False,
        "ooss": False,
    }

    for batch in dataloader:
        outcomes_mask = batch["outcomes_mask"]  # (B, T, O)
        oots_mask = batch["oots_mask"]          # (B, T)
        ooss_mask = batch["ooss_mask"]          # (B,) or (B,1)

        B, T, _ = outcomes_mask.shape

        # Time-wise OOTS
        if oots_mask.dim() == 1:
            oots_time = oots_mask.unsqueeze(-1).expand(-1, T)
        else:
            oots_time = oots_mask

        # Sequence-wise OOSS
        if ooss_mask.dim() == 2 and ooss_mask.shape[1] == 1:
            ooss_seq = ooss_mask[:, 0]
        else:
            ooss_seq = ooss_mask

        cond_ws_wt = (
            (ooss_seq == 0).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 0).unsqueeze(-1)
        )
        cond_valset = (
            (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)
            | (oots_time == 1).unsqueeze(-1)
        )
        cond_ws_oots = (
            (ooss_seq == 0).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 1).unsqueeze(-1)
        )
        cond_wt_ooss = (
            (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 0).unsqueeze(-1)
        )
        cond_oots_ooss = (
            (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)
            & (oots_time == 1).unsqueeze(-1)
        )
        cond_oots = (oots_time == 1).unsqueeze(-1)
        cond_ooss = (ooss_seq == 1).unsqueeze(-1).unsqueeze(-1)

        def subset_has_data(cond):
            subset_mask = cond & outcomes_mask
            return bool(torch.sum(subset_mask) > 0)

        has["ws_wt"] = has["ws_wt"] or subset_has_data(cond_ws_wt)
        has["valset"] = has["valset"] or subset_has_data(cond_valset)
        has["ws_oots"] = has["ws_oots"] or subset_has_data(cond_ws_oots)
        has["wt_ooss"] = has["wt_ooss"] or subset_has_data(cond_wt_ooss)
        has["oots_ooss"] = has["oots_ooss"] or subset_has_data(cond_oots_ooss)
        has["oots"] = has["oots"] or subset_has_data(cond_oots)
        has["ooss"] = has["ooss"] or subset_has_data(cond_ooss)

    return has


def test_sklearn_subset_presence_matches_masks():
    """
    Sanity check: for any subset where masks indicate there is data,
    the sklearn validation metrics should not report -1.0.
    """
    print("\n" + "=" * 80)
    print("CHECKING SKLEARN SUBSET PRESENCE AGAINST MASKS")
    print("=" * 80)

    args = get_default_args()
    args.train_batch_size = 8
    args.val_batch_size = 16
    args.output_dir = "/tmp/sklearn_trainer_subset_check"

    pl.seed_everything(args.seed if hasattr(args, "seed") else 42)
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_from_disk(args.data_dir)
    datasetDict = get_datasetDict(
        train_data=data, val_folds=args.val_folds, test_folds=args.test_folds
    )
    for key in datasetDict:
        datasetDict[key] = datasetDict[key].with_format("torch")

    dataloaderModule = MIDataLoaderModule(args, datasetDict)

    subset_presence = check_subset_presence_from_masks(
        dataloaderModule.val_dataloader()
    )
    print("Subset presence based on masks:", subset_presence)

    model = RidgeForecastModel(alpha=args.weight_decay, random_state=args.seed)
    module = SklearnModule(args, model)
    trainer = SklearnTrainer(output_dir=args.output_dir, logger=None)

    results = trainer.fit(
        module,
        train_dataloader=dataloaderModule.train_dataloader(),
        val_dataloader=dataloaderModule.val_dataloader(),
    )

    val_metrics = results["val"]
    print("Sklearn validation metrics:", val_metrics)

    for subset_name, has_data in subset_presence.items():
        key = f"{subset_name}_mse"
        if has_data:
            if key in val_metrics:
                assert (
                    val_metrics[key] != -1.0
                ), f"{subset_name} has data but {key} is -1.0"
            else:
                raise AssertionError(
                    f"{subset_name} has data but {key} not found in val_metrics"
                )

    print("\n" + "=" * 80)
    print("SKLEARN SUBSET PRESENCE CHECK PASSED")
    print("=" * 80)


def test_autoregressive_sklearn_models():
    """
    Test autoregressive sklearn models with history windowing.
    Compares performance with and without history windowing.
    """
    print("\n" + "=" * 80)
    print("TESTING AUTOREGRESSIVE SKLEARN MODELS")
    print("=" * 80)

    args = get_default_args()
    args.train_batch_size = 8
    args.val_batch_size = 16
    args.output_dir = "/tmp/sklearn_trainer_ar_test"
    
    # Set max_len for autoregressive models (use 7-day history)
    # args.max_len = 7

    pl.seed_everything(args.seed if hasattr(args, "seed") else 42)
    os.makedirs(args.output_dir, exist_ok=True)

    data = load_from_disk(args.data_dir)
    datasetDict = get_datasetDict(
        train_data=data, val_folds=args.val_folds, test_folds=args.test_folds
    )
    for key in datasetDict:
        datasetDict[key] = datasetDict[key].with_format("torch")

    dataloaderModule = MIDataLoaderModule(args, datasetDict)

    results_comparison = {}

    # Test 1: Regular Ridge (no history windowing)
    print("\n" + "-" * 80)
    print("1. Testing Regular Ridge (no history windowing)")
    print("-" * 80)
    model_regular = RidgeForecastModel(alpha=args.weight_decay, random_state=args.seed)
    module_regular = SklearnModule(args, model_regular)
    trainer_regular = SklearnTrainer(output_dir=os.path.join(args.output_dir, "regular"), logger=None)

    results_regular = trainer_regular.fit(
        module_regular,
        train_dataloader=dataloaderModule.train_dataloader(),
        val_dataloader=dataloaderModule.val_dataloader(),
    )
    results_comparison["Regular Ridge"] = results_regular

    print(f"Regular Ridge - Train MSE: {results_regular['train']['mse']:.4f}")
    print(f"Regular Ridge - Val MSE: {results_regular['val']['ws_wt_mse']:.4f}")

    # Test 2: Autoregressive Ridge (with history windowing)
    print("\n" + "-" * 80)
    print(f"2. Testing Autoregressive Ridge (max_len={args.max_len})")
    print("-" * 80)
    model_ar = AutoRegressiveRidge(alpha=args.weight_decay, random_state=args.seed)
    module_ar = SklearnModule(args, model_ar)
    trainer_ar = SklearnTrainer(output_dir=os.path.join(args.output_dir, "autoregressive"), logger=None)

    results_ar = trainer_ar.fit(
        module_ar,
        train_dataloader=dataloaderModule.train_dataloader(),
        val_dataloader=dataloaderModule.val_dataloader(),
    )
    results_comparison["Autoregressive Ridge"] = results_ar

    print(f"Autoregressive Ridge - Train MSE: {results_ar['train']['mse']:.4f}")
    print(f"Autoregressive Ridge - Val MSE: {results_ar['val']['ws_wt_mse']:.4f}")

    # Test 3: Autoregressive Lasso
    print("\n" + "-" * 80)
    print(f"3. Testing Autoregressive Lasso (max_len={args.max_len})")
    print("-" * 80)
    model_ar_lasso = AutoRegressiveLasso(alpha=args.weight_decay, random_state=args.seed, max_iter=1000)
    module_ar_lasso = SklearnModule(args, model_ar_lasso)
    trainer_ar_lasso = SklearnTrainer(output_dir=os.path.join(args.output_dir, "autoregressive_lasso"), logger=None)

    results_ar_lasso = trainer_ar_lasso.fit(
        module_ar_lasso,
        train_dataloader=dataloaderModule.train_dataloader(),
        val_dataloader=dataloaderModule.val_dataloader(),
    )
    results_comparison["Autoregressive Lasso"] = results_ar_lasso

    print(f"Autoregressive Lasso - Train MSE: {results_ar_lasso['train']['mse']:.4f}")
    print(f"Autoregressive Lasso - Val MSE: {results_ar_lasso['val']['ws_wt_mse']:.4f}")

    # Summary comparison
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    for model_name, results in results_comparison.items():
        print(f"\n{model_name}:")
        print(f"  Train MSE: {results['train']['mse']:.4f}")
        if 'ws_wt_mse' in results['val']:
            print(f"  Val MSE (ws_wt): {results['val']['ws_wt_mse']:.4f}")
        if 'valset_mse' in results['val']:
            print(f"  Val MSE (valset): {results['val']['valset_mse']:.4f}")

    print("\n" + "=" * 80)
    print("AUTOREGRESSIVE SKLEARN MODELS TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == '__main__':
    
    # Get default args
    args = get_default_args()
    
    try:
        # test_sklearn_trainer_minimal(args)
        # print("\n" + "=" * 80)
        # print("MINIMAL TEST COMPLETED SUCCESSFULLY")
        # print("=" * 80)

        # Optional: run subset presence check (can be commented out if slow)
        # test_sklearn_subset_presence_matches_masks()
        test_autoregressive_sklearn_models()
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"ERROR: {e}")
        print(f"{'=' * 80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # parser.add_argument('--test', type=str, default='all', 
    #                    choices=['all', 'minimal', 'hparam', 'models'],
    #                    help='Which test to run')
    # args_test = parser.parse_args()
    
    # try:
    #     if args_test.test in ['all', 'minimal']:
    #         test_sklearn_trainer_minimal()
        
    #     if args_test.test in ['all', 'hparam']:
    #         test_hyperparameter_search()
        
    #     if args_test.test in ['all', 'models']:
    #         test_different_models()
        
    #     print("\n" + "=" * 80)
    #     print("ALL TESTS COMPLETED SUCCESSFULLY!")
    #     print("=" * 80)
        
    # except Exception as e:
    #     print(f"\n{'=' * 80}")
    #     print(f"ERROR: {e}")
    #     print(f"{'=' * 80}")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)