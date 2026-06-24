class PortfolioOptimizerError(Exception):
    """Base exception for the Portfolio Optimizer package."""
    pass

class DataDownloadError(PortfolioOptimizerError):
    """Exception raised when downloading historical price or interest rate data fails."""
    pass

class OptimizationError(PortfolioOptimizerError):
    """Exception raised when mathematical optimization fails or is infeasible."""
    pass
