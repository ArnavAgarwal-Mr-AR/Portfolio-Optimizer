import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = BACKEND_ROOT / "config.yaml"
        
        self.config_data = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not read config file {config_path}. Error: {e}. Using defaults.")
        
        # Parse settings
        self.tickers = self._get_tickers()
        
        # Date definitions
        dates_cfg = self.config_data.get("dates", {})
        self.start_date = os.getenv("START_DATE", dates_cfg.get("start_date", "2015-01-01"))
        self.end_date = os.getenv("END_DATE", dates_cfg.get("end_date", "2025-12-31"))
        self.backtest_start_date = dates_cfg.get("backtest_start_date", "2023-01-01")
        
        # Optimization settings
        opt_cfg = self.config_data.get("optimization", {})
        self.default_risk_free_rate = opt_cfg.get("default_risk_free_rate", 0.04)
        self.default_risk_aversion = opt_cfg.get("default_risk_aversion", 1.0)
        self.covariance_method = opt_cfg.get("covariance_method", "shrinkage")
        
        # FRED settings
        fred_cfg = self.config_data.get("fred", {})
        self.fred_api_key = os.getenv("FRED_API_KEY", "")
        # Check if empty string or placeholder
        if not self.fred_api_key or self.fred_api_key.strip() == "":
            self.fred_api_key = None
        self.fred_series_id = fred_cfg.get("series_id", "DGS3MO")
        self.fred_fallback_ticker = fred_cfg.get("fallback_ticker", "^IRX")
        
        # Path settings
        paths_cfg = self.config_data.get("paths", {})
        
        # In Vercel serverless environments, the filesystem is read-only except for /tmp
        is_vercel = os.getenv("VERCEL") == "1"
        if is_vercel:
            self.cache_dir = Path("/tmp") / paths_cfg.get("cache_dir", "data")
            self.reports_dir = Path("/tmp") / paths_cfg.get("reports_dir", "reports")
        else:
            self.cache_dir = PROJECT_ROOT / paths_cfg.get("cache_dir", "data")
            self.reports_dir = PROJECT_ROOT / paths_cfg.get("reports_dir", "reports")
        
        # Ensure directories exist (handling read-only permissions gracefully)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create cache directory {self.cache_dir}. Error: {e}")
            
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create reports directory {self.reports_dir}. Error: {e}")

    def _get_tickers(self):
        env_tickers = os.getenv("PORTFOLIO_TICKERS")
        if env_tickers:
            return [t.strip() for t in env_tickers.split(",") if t.strip()]
        return self.config_data.get("tickers", ["SPY", "TLT", "GLD", "QQQ", "EFA", "VNQ"])

# Instantiate global config
config = Config()
