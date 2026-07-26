from dataclasses import dataclass, field
from typing import Optional, Dict
from enum import Enum

class Vendor(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google"

@dataclass
class TokenPricing:
    """Prices per 1 Million Tokens (USD)"""
    standard_input: float
    output: float
    cached_input: Optional[float] = None
    batch_input: Optional[float] = None
    batch_output: Optional[float] = None

@dataclass
class TieredPricing:
    """For models with context-length-dependent pricing (e.g., Gemini Pro)"""
    threshold_tokens: int
    base_rate: TokenPricing
    over_threshold_rate: TokenPricing

@dataclass
class FreeTierPolicy:
    has_free_tier: bool
    rate_limit_rpm: Optional[int] = None # Requests per minute
    data_used_for_training: bool = True

@dataclass
class AIModel:
    id: str                                  # e.g., "gpt-5.6-sol" or "gemini-3.1-pro"
    name: str                                # e.g., "GPT-5.6 Sol"
    vendor: Vendor
    context_window: int                      # Max input tokens
    pricing: TokenPricing
    tiered_pricing: Optional[TieredPricing] = None
    free_tier: Optional[FreeTierPolicy] = None
    tool_costs: Dict[str, float] = field(default_factory=dict) # e.g., {"web_search_1k": 10.0}

    def calculate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int, 
        cached_tokens: int = 0, 
        is_batch: bool = False
    ) -> float:
        """Calculates exact cost for a request."""
        rates = self.pricing
        
        # Check for tiered threshold (e.g., >200k tokens)
        if self.tiered_pricing and input_tokens > self.tiered_pricing.threshold_tokens:
            rates = self.tiered_pricing.over_threshold_rate

        if is_batch and rates.batch_input and rates.batch_output:
            in_cost = (input_tokens / 1_000_000) * rates.batch_input
            out_cost = (output_tokens / 1_000_000) * rates.batch_output
            return round(in_cost + out_cost, 6)

        # Standard processing
        fresh_tokens = max(0, input_tokens - cached_tokens)
        cached_rate = rates.cached_input if rates.cached_input is not None else rates.standard_input
        
        in_cost = (fresh_tokens / 1_000_000) * rates.standard_input
        cache_cost = (cached_tokens / 1_000_000) * cached_rate
        out_cost = (output_tokens / 1_000_000) * rates.output

        return round(in_cost + cache_cost + out_cost, 6)