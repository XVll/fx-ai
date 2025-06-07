from rich.console import Console

from v2.config import Config

config = Config.load("optuna","optuna-1-foundation")
console = Console()
console.print(config.model_dump())