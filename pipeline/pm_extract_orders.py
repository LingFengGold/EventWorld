"""P0: from the raw Polymarket trade records (Hugging Face SII-WANGZJ/Polymarket_data: data/polymarket/{trades,markets}.parquet) to taker orders
of the long-lived markets.

  1. per-market statistics over all trades; long-lived markets = traded for at least 30 days with at least 5,000 trades (4,176 markets),
     with the question, event title and a keyword topic bucket                              -> out/pm_long_markets.parquet
  2. market metadata deduplicated by id (latest update kept)                                 -> out/pm_deep/markets_dedup.parquet
  3. all trades of the long-lived markets merged into taker orders: one order = the fills of one transaction with the same
     (taker, market, asset, taker direction); YES-side price and signed direction             -> out/pm_deep/long_orders.parquet
pipeline/pm_build_store.py then re-aggregates the orders to block level.
"""
import os, time
import polars as pl

MIN_DAYS, MIN_TRADES = 30, 5000
TOPIC_KEYS = [("nba", "sports"), ("nfl", "sports"), ("mlb", "sports"), ("nhl", "sports"), ("premier league", "sports"), ("champions league", "sports"),
              ("ufc", "sports"), ("tennis", "sports"), ("vs.", "sports"), (" vs ", "sports"), ("bitcoin", "crypto"), ("ethereum", "crypto"),
              ("solana", "crypto"), ("btc", "crypto"), ("eth ", "crypto"), ("election", "politics"), ("president", "politics"), ("trump", "politics"),
              ("senate", "politics"), ("governor", "politics"), ("fed ", "macro"), ("rate cut", "macro"), ("inflation", "macro"), ("cpi", "macro"),
              ("temperature", "weather"), ("oscar", "entertainment"), ("grammy", "entertainment"), ("box office", "entertainment")]


def topic(title):
    t = (title or "").lower()
    for k, v in TOPIC_KEYS:
        if k in t:
            return v
    return "other"


def main():
    t0 = time.time(); os.makedirs("out/pm_deep", exist_ok=True)
    lf = pl.scan_parquet("data/polymarket/trades.parquet")
    stats = (lf.group_by("market_id").agg([
        pl.len().alias("n_trades"), pl.col("usd_amount").sum().alias("volume"),
        pl.col("timestamp").min().alias("t0"), pl.col("timestamp").max().alias("t1"),
        pl.col("usd_amount").median().alias("med_usd"), pl.col("usd_amount").quantile(0.99).alias("p99_usd"),
        pl.col("taker").n_unique().alias("n_takers")]).collect(engine="streaming"))
    stats = stats.with_columns(((pl.col("t1") - pl.col("t0")) / 86400).alias("days"))
    mk_raw = pl.read_parquet("data/polymarket/markets.parquet")
    mk = mk_raw.select(["id", "question", "event_title", "end_date", "created_at", "neg_risk"]).rename({"id": "market_id"}).unique(subset=["market_id"])
    long = stats.join(mk, on="market_id", how="left").filter((pl.col("days") >= MIN_DAYS) & (pl.col("n_trades") >= MIN_TRADES))
    long = long.with_columns(pl.col("event_title").map_elements(topic, return_dtype=pl.Utf8).alias("topic"))
    long.write_parquet("out/pm_long_markets.parquet")
    print(f"long-lived markets (>= {MIN_DAYS}d, >= {MIN_TRADES} trades): {long.height}, trades {long['n_trades'].sum():,} ({time.time() - t0:.0f}s)")

    mk_raw.sort("updated_at").unique(subset=["id"], keep="last").write_parquet("out/pm_deep/markets_dedup.parquet")

    tr = lf.filter(pl.col("market_id").is_in(long["market_id"].to_list())).collect(engine="streaming")
    tr = tr.with_columns([
        pl.when(pl.col("nonusdc_side") == "token1").then(pl.col("price")).otherwise(1 - pl.col("price")).alias("p_yes"),
        (pl.when(pl.col("taker_direction") == "BUY").then(1).otherwise(-1) * pl.when(pl.col("nonusdc_side") == "token1").then(1).otherwise(-1)).cast(pl.Int8).alias("sign"),
    ]).sort(["market_id", "timestamp", "block_number", "log_index"])
    orders = (tr.group_by(["transaction_hash", "taker", "market_id", "asset_id", "taker_direction"]).agg([
        pl.len().alias("n_fills"), pl.col("usd_amount").sum().alias("usd"), pl.col("token_amount").sum().alias("tokens"),
        pl.col("timestamp").first().alias("timestamp"), pl.col("block_number").first().alias("block_number"), pl.col("log_index").min().alias("log_index"),
        pl.col("p_yes").first().alias("p_first"), pl.col("p_yes").last().alias("p_last"), pl.col("sign").first().alias("sign"),
        pl.col("maker").n_unique().alias("n_makers"), pl.col("nonusdc_side").first().alias("nonusdc_side")])
        .with_columns((pl.col("usd") / pl.col("tokens")).alias("vwap_raw"))
        .with_columns(pl.when(pl.col("nonusdc_side") == "token1").then(pl.col("vwap_raw")).otherwise(1 - pl.col("vwap_raw")).alias("vwap_yes"))
        .sort(["market_id", "timestamp", "block_number", "log_index"]))
    orders.write_parquet("out/pm_deep/long_orders.parquet")
    print(f"trades {tr.height:,} -> taker orders {orders.height:,} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
