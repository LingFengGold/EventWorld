"""P1: block-level order store partitioned by market.

Re-aggregates tx-level orders (out/pm_deep/long_orders.parquet) to block level exactly as
the order-merging step does: group by (market_id, taker, block_number, asset_id, taker_direction).
Writes data/processed/polymarket/orders_by_market/market_id=<id>/*.parquet sorted by (block, log_index)
with an integer row position `idx`, integer wallet ids (wallets.parquet) and a per-market index (markets.parquet).
"""
import time, sys, os, shutil
import polars as pl, numpy as np

ROOT = "data/processed/polymarket"
STORE = f"{ROOT}/orders_by_market"


def main():
    t0 = time.time()
    orders = pl.read_parquet("out/pm_deep/long_orders.parquet")
    print("tx-level orders:", orders.height, "elapsed %.0fs" % (time.time() - t0))
    orders = (orders.sort(["market_id", "timestamp", "block_number", "log_index"])
              .group_by(["market_id", "taker", "block_number", "asset_id", "taker_direction"], maintain_order=True)
              .agg([pl.col("n_fills").sum(), pl.col("usd").sum(), pl.col("tokens").sum(), pl.col("timestamp").first(),
                    pl.col("log_index").min(), pl.col("sign").first(), pl.col("nonusdc_side").first(),
                    pl.col("transaction_hash").n_unique().alias("n_tx")])
              .with_columns((pl.col("usd") / pl.col("tokens")).alias("vwap_raw"))
              .with_columns(pl.when(pl.col("nonusdc_side") == "token1").then(pl.col("vwap_raw")).otherwise(1 - pl.col("vwap_raw")).alias("p")))
    print("block-level orders:", orders.height, "elapsed %.0fs" % (time.time() - t0))
    # wallet ids: deterministic (sorted address), 1-based; 0 reserved for "none"
    wallets = orders.group_by("taker").agg(pl.len().alias("n_orders"), pl.col("usd").sum().alias("usd"), pl.col("market_id").n_unique().alias("n_markets")).sort("taker")
    wallets = wallets.with_row_index("wallet", offset=1).with_columns(pl.col("wallet").cast(pl.UInt32))
    wallets.select("wallet", pl.col("taker").alias("address"), "n_orders", "usd", "n_markets").write_parquet(f"{ROOT}/wallets.parquet")
    print("wallets:", wallets.height)
    orders = (orders.join(wallets.select("taker", "wallet"), on="taker", how="left")
              .sort(["market_id", "block_number", "log_index"])
              .with_columns(pl.int_range(pl.len()).over("market_id").cast(pl.UInt32).alias("idx"))
              .with_columns(pl.col("timestamp").cast(pl.Int64).alias("ts_raw"))
              # block order is canonical (block-strict history); 0.04% of orders carry a timestamp up to ~200s out of block
              # order (Polygon clock skew), so the effective time is the running max of ts along block order per market
              .with_columns(pl.col("ts_raw").cum_max().over("market_id").alias("ts"))
              .select("market_id", "idx", pl.col("block_number").cast(pl.UInt64).alias("block"), "ts", "ts_raw",
                      "log_index", "wallet", "sign", "usd", "tokens", "p", "n_fills", "n_tx", pl.col("asset_id"), pl.col("taker_direction")))
    # sanity: block order == time order within market
    bad = orders.select((pl.col("ts_raw").diff().over("market_id") < 0).sum().alias("ts_decreasing")).item()
    adj = orders.select((pl.col("ts") != pl.col("ts_raw")).sum()).item()
    print("orders with raw ts decreasing along block order:", bad, "; orders whose effective ts was lifted:", adj,
          "; max lift (s):", orders.select((pl.col("ts") - pl.col("ts_raw")).max()).item())
    assert orders.select((pl.col("ts").diff().over("market_id") < 0).sum()).item() == 0
    if os.path.exists(STORE):
        shutil.rmtree(STORE)
    orders.write_parquet(STORE, partition_by="market_id")
    # market index
    mk = pl.read_parquet("out/pm_long_markets.parquet")
    dd = pl.read_parquet("out/pm_deep/markets_dedup.parquet").select(pl.col("id").alias("market_id"), "event_id", "event_slug", "slug", "answer1", "answer2", "outcome_prices", "closed")
    idx = (orders.group_by("market_id").agg(pl.len().alias("n_orders"), pl.col("ts").min().alias("t_first"), pl.col("ts").max().alias("t_last"),
                                            pl.col("block").min().alias("b_first"), pl.col("block").max().alias("b_last"), pl.col("wallet").n_unique().alias("n_wallets"))
           .join(mk.select("market_id", "question", "event_title", "end_date", "created_at", "neg_risk", "topic", "n_trades", "volume", "days"), on="market_id", how="left")
           .join(dd, on="market_id", how="left").sort("t_first"))
    idx.write_parquet(f"{ROOT}/markets.parquet")
    print("markets:", idx.height, " event_id null:", idx["event_id"].null_count())
    print(f"store written: {orders.height} rows, {idx.height} markets, elapsed {time.time()-t0:.0f}s")
    os.makedirs("out/build", exist_ok=True)
    with open("out/build/pm_store.md", "w") as f:
        f.write("# P1 Polymarket order store\n\n")
        f.write(f"- tx-level orders in: 80.8M -> block-level orders: {orders.height:,}\n- markets: {idx.height}; wallets: {wallets.height:,}\n")
        f.write(f"- layout: `{STORE}/market_id=<id>/` sorted by (block, log_index), `idx` = row position; `wallets.parquet` maps wallet id -> address; `markets.parquet` per-market index\n")
        f.write(f"- `ts` = effective time = running max of raw timestamp along block order ({adj:,} orders lifted, {bad:,} raw inversions); `ts_raw` kept\n")
        f.write(f"- direction: `sign` = +1 taker buys YES, -1 sells YES (taker_direction x token side); `p` = YES-view vwap\n- runtime {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
