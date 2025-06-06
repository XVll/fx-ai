#!/usr/bin/env python3
"""Test if Captum analysis runs correctly."""

import sys
import os
import logging
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_captum_direct():
    """Test Captum analysis directly."""
    
    from config.loader import load_config
    from feature.attribution.captum_attribution import AttributionConfig, CaptumAttributionAnalyzer
    from ai.transformer import MultiBranchTransformer
    
    # Load config
    config = load_config('quick')
    
    # Create model
    model_config = config.model
    model = MultiBranchTransformer(
        model_config=model_config,
        device="cpu",
        logger=logger
    )
    model.eval()
    
    # Create Captum analyzer
    attr_config = AttributionConfig(
        methods=["saliency"],
        n_steps=5,
        analyze_branches=False,
        analyze_fusion=False,
        analyze_actions=True,
        save_visualizations=False,
    )
    
    analyzer = CaptumAttributionAnalyzer(config=attr_config, model=model)
    
    # Create dummy state
    state = {
        "hf": torch.randn(1, model_config.hf_seq_len, model_config.hf_feat_dim),
        "mf": torch.randn(1, model_config.mf_seq_len, model_config.mf_feat_dim),
        "lf": torch.randn(1, model_config.lf_seq_len, model_config.lf_feat_dim),
        "portfolio": torch.randn(1, model_config.portfolio_seq_len, model_config.portfolio_feat_dim),
    }
    
    logger.info("Running Captum analysis...")
    
    # Get model output to use as target
    with torch.no_grad():
        outputs = model(state)
        logger.info(f"Model output type: {type(outputs)}")
        if isinstance(outputs, tuple):
            logger.info(f"Output tuple length: {len(outputs)}")
            for i, out in enumerate(outputs):
                logger.info(f"  Output {i}: {type(out)} shape: {out.shape if hasattr(out, 'shape') else 'N/A'}")
            action_logits = outputs[0][0] if isinstance(outputs[0], tuple) else outputs[0]
        else:
            action_logits = outputs
        target_action = action_logits.argmax(dim=-1)
    
    try:
        results = analyzer.analyze_sample(state, target_action=target_action)
        
        logger.info(f"Analysis complete!")
        logger.info(f"Results keys: {list(results.keys())}")
        
        if 'attributions' in results:
            logger.info(f"Methods analyzed: {list(results['attributions'].keys())}")
        else:
            logger.info(f"No attributions in results")
        
        # Check top attributions
        if results.get('top_attributions'):
            logger.info(f"\nTop attributions:")
            for method, data in results['top_attributions'].items():
                logger.info(f"\n{method}:")
                if isinstance(data, dict):
                    for branch, features in data.items():
                        if isinstance(features, list):
                            logger.info(f"  {branch}:")
                            for feat in features[:3]:
                                if isinstance(feat, dict):
                                    logger.info(f"    - {feat.get('name', 'unknown')}: {feat.get('importance', 0):.4f}")
        
        logger.info("\n✅ Captum analysis successful!")
        
    except Exception as e:
        logger.error(f"❌ Captum analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_captum_direct()