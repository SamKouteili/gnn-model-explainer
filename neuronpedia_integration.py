"""
Neuronpedia Integration for Semantic Feature Analysis
Adds interpretable descriptions, top logits, and examples to feature analysis
"""

import requests
import json
import os
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass
import time
import logging

@dataclass
class FeatureInterpretation:
    """Container for Neuronpedia feature interpretation data"""
    feature_id: int
    layer: int
    description: str
    explanation: Optional[str]
    top_logits: List[Dict[str, Any]]
    bottom_logits: List[Dict[str, Any]]
    max_activating_examples: List[Dict[str, Any]]
    activation_stats: Dict[str, float]
    auto_interp_score: Optional[float]

class NeuronpediaClient:
    """Client for accessing Neuronpedia API and data"""
    
    def __init__(self, api_key: Optional[str] = None, model_id: str = "gemma-2-2b"):
        self.api_key = api_key or os.getenv("NEURONPEDIA_API_KEY")
        self.base_url = "https://neuronpedia.org/api"
        self.model_id = model_id
        self.session = requests.Session()
        self.cache = {}  # Simple cache to avoid duplicate requests
        
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def get_gemmascope_layer_id(self, layer: int, transcoder_type: str = "res") -> str:
        """Convert layer number to GemmaScope layer identifier"""
        return f"{layer}-gemmascope-{transcoder_type}-16k"
    
    def get_feature_interpretation(self, layer: int, feature_idx: int) -> Optional[FeatureInterpretation]:
        """Get comprehensive feature interpretation from Neuronpedia"""
        cache_key = f"{layer}_{feature_idx}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        layer_id = self.get_gemmascope_layer_id(layer)
        url = f"{self.base_url}/feature/{self.model_id}/{layer_id}/{feature_idx}"
        
        try:
            self.logger.info(f"Fetching interpretation for Layer {layer}, Feature {feature_idx}")
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                interpretation = self._parse_feature_data(data, layer, feature_idx)
                self.cache[cache_key] = interpretation
                return interpretation
            elif response.status_code == 404:
                self.logger.warning(f"Feature not found: Layer {layer}, Feature {feature_idx}")
                return None
            else:
                self.logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except requests.RequestException as e:
            self.logger.error(f"Request failed for Layer {layer}, Feature {feature_idx}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
    
    def _parse_feature_data(self, data: Dict, layer: int, feature_idx: int) -> FeatureInterpretation:
        """Parse Neuronpedia API response into structured interpretation"""
        
        # Extract basic information
        description = data.get("description", "No description available")
        explanation = data.get("explanation", {}).get("text", None)
        auto_interp_score = data.get("explanation", {}).get("score", None)
        
        # Extract logit information
        top_logits = []
        bottom_logits = []
        
        logits_data = data.get("logits", {})
        if "top_logits" in logits_data:
            for logit in logits_data["top_logits"][:10]:  # Top 10
                top_logits.append({
                    "token": logit.get("token", ""),
                    "logit_diff": logit.get("logit_diff", 0.0),
                    "rank": logit.get("rank", 0)
                })
        
        if "bottom_logits" in logits_data:
            for logit in logits_data["bottom_logits"][:10]:  # Bottom 10
                bottom_logits.append({
                    "token": logit.get("token", ""),
                    "logit_diff": logit.get("logit_diff", 0.0),
                    "rank": logit.get("rank", 0)
                })
        
        # Extract max activating examples
        max_examples = []
        examples_data = data.get("activations", {}).get("examples", [])
        for example in examples_data[:5]:  # Top 5 examples
            max_examples.append({
                "text": example.get("text", ""),
                "activation": example.get("activation", 0.0),
                "tokens": example.get("tokens", [])
            })
        
        # Extract activation statistics
        act_stats = data.get("activations", {}).get("stats", {})
        activation_stats = {
            "mean": act_stats.get("mean", 0.0),
            "std": act_stats.get("std", 0.0),
            "max": act_stats.get("max", 0.0),
            "frequency": act_stats.get("frequency", 0.0)
        }
        
        return FeatureInterpretation(
            feature_id=feature_idx,
            layer=layer,
            description=description,
            explanation=explanation,
            top_logits=top_logits,
            bottom_logits=bottom_logits,
            max_activating_examples=max_examples,
            activation_stats=activation_stats,
            auto_interp_score=auto_interp_score
        )
    
    def get_bulk_interpretations(self, feature_list: List[tuple], delay: float = 0.1) -> Dict[tuple, FeatureInterpretation]:
        """Get interpretations for multiple features with rate limiting"""
        interpretations = {}
        
        for i, (layer, feature_idx) in enumerate(feature_list):
            if i > 0:  # Rate limiting
                time.sleep(delay)
            
            interp = self.get_feature_interpretation(layer, feature_idx)
            if interp:
                interpretations[(layer, feature_idx)] = interp
        
        return interpretations

def reverse_feature_normalization(processed_feature_id: float, feature_type: str) -> int:
    """Reverse the feature normalization done in semantic converter"""
    if feature_type == 'cross layer transcoder':
        # Was normalized as: feature_val / 1000.0 (no more clamping)
        # Reverse the normalization
        return int(processed_feature_id * 1000)
    elif feature_type == 'embedding':
        # Was kept as-is for small values
        return int(processed_feature_id)
    elif feature_type == 'logit':
        # Was normalized as: feature_val / 10000.0
        return int(processed_feature_id * 10000)
    else:
        return None

def enrich_analysis_with_neuronpedia(df: pd.DataFrame, client: NeuronpediaClient) -> pd.DataFrame:
    """Enrich node analysis DataFrame with Neuronpedia interpretations"""
    
    # Debug: Print available columns
    print(f"Available DataFrame columns: {list(df.columns)}")
    
    # Check what node types we have
    if 'node_type' in df.columns:
        print(f"Node types found: {df['node_type'].value_counts().to_dict()}")
    else:
        print("WARNING: 'node_type' column not found!")
        return df
    
    # We'll use the processed feature information and reverse the normalization
    # Check for layer column - could be 'layer_feat' or 'original_layer'
    if 'layer_feat' in df.columns:
        layer_col = 'layer_feat'
    elif 'original_layer' in df.columns:
        layer_col = 'original_layer'
    else:
        print(f"WARNING: No layer column found. Available: {list(df.columns)}")
        return df
    
    if 'processed_feature_id' not in df.columns:
        print(f"WARNING: processed_feature_id column not found. Available: {list(df.columns)}")
        return df
        
    processed_feature_col = 'processed_feature_id'
    
    # Extract transcoder features from analysis
    transcoder_mask = (df['node_type'] == 'cross layer transcoder')
    if transcoder_mask.sum() == 0:
        # Try alternative node type names
        transcoder_mask = (df['node_type'] == 'transcoder')
    
    transcoder_features = df[
        transcoder_mask & 
        (df[processed_feature_col].notna()) & 
        (df[layer_col].notna())
    ].copy()
    
    print(f"Found {len(transcoder_features)} transcoder features for enrichment")
    
    if transcoder_features.empty:
        print("No transcoder features found for enrichment")
        print("Sample of available data:")
        print(df[['node_type', processed_feature_col, layer_col]].head())
        return df
    
    # Prepare feature list for bulk lookup by reversing normalization
    feature_list = []
    for _, row in transcoder_features.head(20).iterrows():  # Limit to top 20 for demo
        try:
            layer = int(row[layer_col])
            processed_feature_id = float(row[processed_feature_col])
            node_type = row['node_type']
            
            # Reverse the normalization to get original feature ID
            original_feature_id = reverse_feature_normalization(processed_feature_id, node_type)
            
            if original_feature_id is not None:
                feature_list.append((layer, original_feature_id))
                print(f"Row {row.name}: Layer {layer}, Processed {processed_feature_id} -> Original {original_feature_id}")
            else:
                print(f"Row {row.name}: Skipping (processed_feature_id={processed_feature_id} was clamped)")
                
        except (ValueError, TypeError) as e:
            print(f"Error processing row: layer={row.get(layer_col)}, processed_feature_id={row.get(processed_feature_col)}, error={e}")
            continue
    
    print(f"Fetching interpretations for {len(feature_list)} features...")
    
    # Get interpretations
    interpretations = client.get_bulk_interpretations(feature_list)
    
    # Add interpretation columns to DataFrame
    df['neuronpedia_description'] = None
    df['neuronpedia_explanation'] = None
    df['top_promoted_tokens'] = None
    df['top_suppressed_tokens'] = None
    df['max_activating_text'] = None
    df['interpretation_score'] = None
    
    # Populate interpretation data
    for _, row in df.iterrows():
        try:
            # Check if this is a transcoder feature with available data
            is_transcoder = (row['node_type'] in ['cross layer transcoder', 'transcoder'])
            has_feature_data = (pd.notna(row.get(processed_feature_col)) and pd.notna(row.get(layer_col)))
            
            if is_transcoder and has_feature_data:
                layer = int(row[layer_col])
                processed_feature_id = float(row[processed_feature_col])
                
                # Reverse normalization to get original feature ID
                original_feature_id = reverse_feature_normalization(processed_feature_id, row['node_type'])
                
                if original_feature_id is not None:
                    key = (layer, original_feature_id)
                    
                    if key in interpretations:
                        interp = interpretations[key]
                        df.at[row.name, 'neuronpedia_description'] = interp.description
                        df.at[row.name, 'neuronpedia_explanation'] = interp.explanation
                        df.at[row.name, 'interpretation_score'] = interp.auto_interp_score
                        
                        # Format top promoted/suppressed tokens
                        if interp.top_logits:
                            top_tokens = [f"'{t['token']}' ({t['logit_diff']:.2f})" 
                                        for t in interp.top_logits[:5]]
                            df.at[row.name, 'top_promoted_tokens'] = "; ".join(top_tokens)
                        
                        if interp.bottom_logits:
                            bottom_tokens = [f"'{t['token']}' ({t['logit_diff']:.2f})" 
                                           for t in interp.bottom_logits[:5]]
                            df.at[row.name, 'top_suppressed_tokens'] = "; ".join(bottom_tokens)
                        
                        # Add example text
                        if interp.max_activating_examples:
                            example_text = interp.max_activating_examples[0]['text'][:100] + "..."
                            df.at[row.name, 'max_activating_text'] = example_text
                        
        except (ValueError, TypeError, KeyError) as e:
            print(f"Error processing interpretation for row {row.name}: {e}")
            continue
    
    return df

def create_interpretation_report(df: pd.DataFrame, output_file: str = "feature_interpretations.html"):
    """Create an HTML report with feature interpretations"""
    
    # Filter to features with interpretations
    interpreted_features = df[df['neuronpedia_description'].notna()].copy()
    
    if interpreted_features.empty:
        print("No features have Neuronpedia interpretations")
        return
    
    # Sort by influence
    interpreted_features = interpreted_features.sort_values('total_influence', ascending=False)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Feature Interpretation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .feature {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
            .feature-header {{ font-weight: bold; color: #333; margin-bottom: 10px; }}
            .influence {{ color: #007acc; font-weight: bold; }}
            .description {{ background-color: #f9f9f9; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            .tokens {{ font-family: monospace; background-color: #f0f0f0; padding: 5px; margin: 5px 0; }}
            .example {{ font-style: italic; color: #666; margin: 5px 0; }}
        </style>
    </head>
    <body>
        <h1>GNN Explainer Feature Interpretation Report</h1>
        <p>Generated from Neuronpedia data for {len(interpreted_features)} transcoder features</p>
    """
    
    for _, feature in interpreted_features.iterrows():
        html_content += f"""
        <div class="feature">
            <div class="feature-header">
                Node {feature['node_idx']}: Layer {feature.get('original_layer', 'N/A')} Feature {feature.get('feature_id', 'N/A')}
                <span class="influence">(Influence: {feature['total_influence']:.4f})</span>
            </div>
            
            <div class="description">
                <strong>Description:</strong> {feature.get('neuronpedia_description', 'No description')}
            </div>
            
            {f'<div><strong>Auto-interpretation:</strong> {feature["neuronpedia_explanation"]}</div>' if pd.notna(feature.get('neuronpedia_explanation')) else ''}
            
            {f'<div class="tokens"><strong>Promotes tokens:</strong> {feature["top_promoted_tokens"]}</div>' if pd.notna(feature.get('top_promoted_tokens')) else ''}
            
            {f'<div class="tokens"><strong>Suppresses tokens:</strong> {feature["top_suppressed_tokens"]}</div>' if pd.notna(feature.get('top_suppressed_tokens')) else ''}
            
            {f'<div class="example"><strong>Example activation:</strong> {feature["max_activating_text"]}</div>' if pd.notna(feature.get('max_activating_text')) else ''}
            
            <div><strong>Node ID:</strong> {feature.get('node_id', 'N/A')}</div>
            <div><strong>Connections:</strong> {feature.get('total_connections', 'N/A')}</div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Interpretation report saved to {output_file}")

# Usage example
if __name__ == "__main__":
    # Example usage
    client = NeuronpediaClient()  # Set NEURONPEDIA_API_KEY environment variable
    
    # Test single feature lookup
    interp = client.get_feature_interpretation(layer=0, feature_idx=161)
    if interp:
        print(f"Feature 161 (Layer 0): {interp.description}")
        print(f"Explanation: {interp.explanation}")
        print(f"Top tokens: {[t['token'] for t in interp.top_logits[:3]]}")