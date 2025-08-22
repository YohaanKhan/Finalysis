import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import random
from dotenv import load_dotenv
from pandas_datareader import data as pdr
from sentiment import get_final_sentiment

# We faced errors, and it turns out that XGBoost requires specific columns to be present in the dataset IN ORDER to make predictions.
# This is the expected structure of the dataset. so we defined it here to ensure that the ingestion pipeline produces the correct columns.
expected_columns = [
    'Current Ratio', 'Long-term Debt / Capital', 'Debt/Equity Ratio', 'Gross Margin',
    'Operating Margin', 'EBIT Margin', 'EBITDA Margin', 'Pre-Tax Profit Margin',
    'Net Profit Margin', 'Asset Turnover', 'ROE - Return On Equity',
    'Return On Tangible Equity', 'ROA - Return On Assets', 'ROI - Return On Investment',
    'Operating Cash Flow Per Share', 'Free Cash Flow Per Share',
    '10Y_Treasury_Yield', 'BAA_Spread', 'Unemployment_Rate',
    'Industrial_Production', 'CPI_Inflation', 'Sentiment_Score'
]


def ingest_yfinance_data(ticker: str) -> None:
    """
    Retrieves and processes financial data for a specified stock ticker using yfinance,
    calculates financial ratios, and saves them to CSV.
    """

    load_dotenv()

    try:
        company = yf.Ticker(ticker)

        # Financial statements
        balance_sheet = company.balance_sheet.transpose().iloc[0]
        income_statement = company.income_stmt.transpose().iloc[0]
        cash_flow = company.cashflow.transpose().iloc[0]

        # Core values
        total_revenue = float(income_statement.get('Total Revenue', 0) or 0)
        total_equity = float(balance_sheet.get('Stockholders Equity', 0) or 1)
        total_assets = float(balance_sheet.get('Total Assets', 0) or 1)
        shares_outstanding = float(company.info.get(
            'sharesOutstanding',
            income_statement.get('Basic Average Shares', 1) or 1
        ))
        total_current_liabilities = float(balance_sheet.get('Current Liabilities', 0) or 1)
        total_debt = float(balance_sheet.get('Total Debt', 0) or 0)
        total_current_assets = float(balance_sheet.get('Current Assets', 0) or 0)
        tangible_equity = total_equity - float(balance_sheet.get("Goodwill", 0))  # tangible equity

        # Feature mapping (ONLY financial ratios + sentiment, no macros here!)
        # The macros are added later in the pipeline
        features = {
            "Rating Date": datetime.now(),

            "Current Ratio": float(total_current_assets / total_current_liabilities)
            if total_current_liabilities != 0 else 0.0,

            "Long-term Debt / Capital": float(balance_sheet.get('Long Term Debt', 0)),

            "Debt/Equity Ratio": float(total_debt / total_equity) if total_equity != 0 else 0.0,

            "Gross Margin": float((income_statement.get('Gross Profit', 0) / total_revenue) * 100)
            if total_revenue != 0 else 0.0,

            "Operating Margin": float((income_statement.get('Operating Income',
                                income_statement.get('Operating Income Or Loss', 0)) / total_revenue) * 100)
            if total_revenue != 0 else 0.0,

            "EBIT Margin": float((income_statement.get('EBIT',
                                income_statement.get('Ebit', 0)) / total_revenue) * 100)
            if total_revenue != 0 else 0.0,

            "EBITDA Margin": float(((income_statement.get('EBIT',
                                 income_statement.get('Ebit', 0)) +
                                 income_statement.get('Reconciled Depreciation',
                                 income_statement.get('Depreciation And Amortization', 0))) / total_revenue) * 100)
            if total_revenue != 0 else 0.0,

            "Pre-Tax Profit Margin": float((income_statement.get('Pretax Income',
                                      income_statement.get('Income Before Tax', 0)) / total_revenue) * 100)
            if total_revenue != 0 else 0.0,

            "Net Profit Margin": float((income_statement.get('Net Income', 0) / total_revenue) * 100)
            if total_revenue != 0 else 0.0,

            "Asset Turnover": float(total_revenue / total_assets) if total_assets != 0 else 0.0,

            "ROE - Return On Equity": float((income_statement.get('Net Income', 0) / total_equity) * 100)
            if total_equity != 0 else 0.0,

            "Return On Tangible Equity": float((income_statement.get('Net Income', 0) / tangible_equity) * 100)
            if tangible_equity != 0 else 0.0,

            "ROA - Return On Assets": float((income_statement.get('Net Income', 0) / total_assets) * 100)
            if total_assets != 0 else 0.0,

            "ROI - Return On Investment": float((income_statement.get('Net Income', 0) / total_assets) * 100)
            if total_assets != 0 else 0.0,  # simplified as ROA proxy

            "Operating Cash Flow Per Share": float(cash_flow.get('Operating Cash Flow', 0) / shares_outstanding)
            if shares_outstanding != 0 else 0.0,

            "Free Cash Flow Per Share": float(
                (cash_flow.get('Operating Cash Flow', 0) -
                 cash_flow.get('Purchase Of PPE', cash_flow.get('Capital Expenditure', 0))) / shares_outstanding
            ) if shares_outstanding != 0 else 0.0,

            "Sentiment_Score": get_final_sentiment(ticker)
        }

        # Save
        output_path = f"data/{company.info['symbol']}_financials.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        pd.DataFrame([features]).to_csv(
            output_path,
            mode="a",
            index=False,
            header=not os.path.exists(output_path)
        )

    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")



def add_macro_data_from_fred(
    company_csv,
    start_date="2000-01-01",
    end_date="2025-12-31",
    save_as="structured_data_company.csv"
) -> pd.DataFrame:
    """
    Adds macroeconomic data from FRED to company dataset.
    """

    # Macro series
    # This contains the top 5 possible macroeconomic indicators that could have a effect on the credit rating.
    macro_series = {
        "10Y_Treasury_Yield": "DGS10",
        "BAA_Spread": "BAA10YM",
        "Unemployment_Rate": "UNRATE",
        "Industrial_Production": "INDPRO",
        "CPI_Inflation": "CPIAUCSL"
    }

    macro_df = pd.DataFrame()

    # Fetch each series and merge
    for col, fred_code in macro_series.items():
        print(f"Fetching {col} from FRED...")
        series = pdr.DataReader(fred_code, "fred", start_date, end_date)
        series.rename(columns={fred_code: col}, inplace=True)

        if macro_df.empty:
            macro_df = series
        else:
            macro_df = macro_df.join(series, how="outer")

    # We remake the DATE index and reset it to a column
    macro_df = macro_df.reset_index().rename(columns={"DATE": "DATE"})

    # Fill missing values with forward fill
    macro_df = macro_df.fillna(method="ffill")

    # Finally, we sort the dataframe by date
    macro_df = macro_df.sort_values("DATE")

    # Importing the company CSV that contains the financials
    df = pd.read_csv(company_csv)
    df["Rating Date"] = pd.to_datetime(df["Rating Date"])
    df = df.sort_values("Rating Date")

    # This is a special merge that merges the macro data with the company data
    # We use merge_asof to align the dates, which is useful for time series data
    # This will merge the macro data with the company data based on the closest date before or on the company's rating date
    merged = pd.merge_asof(
        df,
        macro_df,
        left_on="Rating Date",
        right_on="DATE",
        direction="backward"
    ).drop(columns=["DATE"])

    return merged


def run_pipeline(ticker: str):
    """
    Full pipeline:
    1. Ingest financials
    2. Merge with FRED macros
    """

    ingest_yfinance_data(ticker)
    company_csv = f"data/{ticker}_financials.csv"

    final_df = add_macro_data_from_fred(
        company_csv=company_csv,
        start_date="2000-01-01",
        end_date="2025-12-31",
    )
    
    # Ensure the final DataFrame has the expected columns in the correct order
    final_df = final_df[expected_columns]
    final_df.to_csv(f"data/{ticker}_structured.csv", index=False)

    return final_df


# Example
# df = run_pipeline("IBM")
# print(df.head())
