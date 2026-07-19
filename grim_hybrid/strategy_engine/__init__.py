"""
strategy_engine
===============

The single source of truth for all ICT / Silver Bullet trading logic.

Public entrypoint:
    from strategy_engine.engine import StrategyEngine, SetupResult
    engine = StrategyEngine(config)
    setup = engine.analyze(market_data)          # SetupResult
    features = engine.build_features(setup)      # dict[str, float]

Every downstream component (live server, collector, trainer, future
backtester) must go through StrategyEngine.  No trading logic is allowed
to live anywhere else - that is the whole point of this refactor.
"""

from strategy_engine.engine import StrategyEngine, SetupResult, MarketData

__all__ = ["StrategyEngine", "SetupResult", "MarketData"]
