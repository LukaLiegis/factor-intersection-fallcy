import numpy as np
import polars as pl
from scipy import stats
import matplotlib.pyplot as plt


def simulate_stock_universe(
        n_stocks: int = 1000,
        n_factors: int = 3,
        true_factor_returns: float = 0.05,
        idio_vol: float = 0.30,
) -> pl.DataFrame:

    factor_data = {}
    factor_exposures = []

    for i in range(n_factors):
        raw_exposure = np.random.randn(n_stocks)

        normalized = (raw_exposure - raw_exposure.mean()) / raw_exposure.std()
        factor_data[f'factor_{i + 1}'] = normalized
        factor_exposures.append(normalized)

    factor_exposures_array = np.array(factor_exposures).T
    true_returns = (factor_exposures_array.sum(axis = 1) * true_factor_returns +
                    np.random.randn(n_stocks) * idio_vol)

    factor_data[f'stock_id'] = np.arange(n_stocks)
    factor_data[f'return'] = true_returns

    df = pl.DataFrame(factor_data)

    return df


def create_intersected_portfolio(
        df: pl.DataFrame,
        threshold: float = 1.0,
) -> pl.DataFrame:

    factor_cols = sorted(df.select(pl.exclude('stock_id', 'return')).columns)

    condition = pl.col(factor_cols[0]) > threshold
    for col in factor_cols[1:]:
        condition = condition & (pl.col(col) > threshold)

    selected = df.filter(condition)

    return selected


def create_union_portfolio(
        df: pl.DataFrame,
        top_pct: float = 0.03,
) -> pl.DataFrame:

    factor_cols = sorted(df.select(pl.exclude('stock_id', 'return')).columns)

    df_with_avg = df.with_columns(
        pl.mean_horizontal(factor_cols).alias('avg_score')
    )

    threshold = df_with_avg['avg_score'].quantile(1 - top_pct)
    selected = df_with_avg.filter(pl.col('avg_score') > threshold)

    return selected


def main(
        n_sims: int = 1000
):
    results = {
        'intersection_returns': [],
        'union_returns': [],
        'intersection_sharpe': [],
        'union_sharpe': [],
    }

    for sim in range(n_sims):
        df = simulate_stock_universe()

        int_portfolio = create_intersected_portfolio(df)

        union_portfolio = create_union_portfolio(df)

        int_return = int_portfolio['return'].mean()
        int_vol = int_portfolio['return'].std()
        int_sharpe = int_return / int_vol

        uni_return = union_portfolio['return'].mean()
        uni_vol = union_portfolio['return'].std()
        uni_sharpe = uni_return / uni_vol

        results['intersection_returns'].append(int_return)
        results['union_returns'].append(uni_return)
        results['intersection_sharpe'].append(int_sharpe)
        results['union_sharpe'].append(uni_sharpe)

    return results


if __name__ == "__main__":

    results = main()

    print("\nSHARPE RATIO:")
    print(f"Intersection: {np.mean(results['intersection_sharpe']):.3f}")
    print(f"Union:        {np.mean(results['union_sharpe']):.3f}")
    print(f"Union is {(np.mean(results['union_sharpe']) / np.mean(results['intersection_sharpe']) - 1) * 100:.0f}% better!")
