// SPDX-License-Identifier: MIT
//
// polymarket-faa-attribution-02 entry (Node 22, standard library only).
//
// Model, per market, from that market's own minute prices up to the checkpoint:
// the centre is the last price (persistence). The spread around it is the
// market's own recent behaviour, not a global constant: take the last 960
// minutes of prices, measure the empirical quantiles of the h-minute price
// changes at anchor horizons (5, 15, 60, 240 minutes), recentre them on their
// own median so the median row stays on the last price, interpolate in
// sqrt(h) between anchors, and blend 90% of that with 10% of the house
// baseline's normal random-walk spread (z * 0.0025 * sqrt(h)) as a floor for
// markets whose recent window was eerily quiet. Quiet markets get narrow
// bands, jumpy markets wide ones, and the asymmetry of the market's real
// moves shapes the tails. Deterministic: no clock, no randomness, no network.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];
/** The house baseline's per-minute spread, the blend's parametric part. */
const SIGMA0 = 0.0025;
/** Minutes of recent history the empirical distribution is measured on. */
const WINDOW = 960;
/** Anchor horizons (minutes) for the empirical change distribution. */
const ANCHORS = [5, 15, 60, 240];
/** Weight of the empirical deviations against the parametric ones. */
const EMP_W = 0.9;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

/** x clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** The tau-quantile of an ascending-sorted array, linear interpolation. */
function quantileSorted(s: number[], tau: number): number {
  if (s.length === 1) return s[0] as number;
  const pos = tau * (s.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(s.length - 1, lo + 1);
  const frac = pos - lo;
  return (s[lo] as number) * (1 - frac) + (s[hi] as number) * frac;
}

/** Centred empirical deviation quantiles at anchor horizons: [anchor, devs]. */
function anchorDeviations(prices: number[], horizon: number): Array<[number, number[]]> {
  const n = prices.length;
  const out: Array<[number, number[]]> = [];
  for (const H of ANCHORS) {
    if (H > horizon || n - H < 20) continue;
    const stride = Math.max(1, Math.floor(H / 3));
    const changes: number[] = [];
    for (let i = 0; i + H < n; i += stride) changes.push((prices[i + H] as number) - (prices[i] as number));
    if (changes.length < 20) continue;
    changes.sort((u, v) => u - v);
    const devs = QUANTILES.map((t) => quantileSorted(changes, t));
    const mid = devs[2] as number;
    out.push([H, devs.map((d) => d - mid)]);
  }
  return out;
}

/** Empirical deviation at horizon h: interpolation in sqrt(h) between anchors. */
function empiricalDev(anchors: Array<[number, number[]]>, tauIdx: number, h: number): number {
  const first = anchors[0] as [number, number[]];
  const last = anchors[anchors.length - 1] as [number, number[]];
  if (h <= first[0]) return (first[1][tauIdx] as number) * Math.sqrt(h / first[0]);
  if (h >= last[0]) return (last[1][tauIdx] as number) * Math.sqrt(h / last[0]);
  for (let k = 0; k + 1 < anchors.length; k++) {
    const a = anchors[k] as [number, number[]];
    const b = anchors[k + 1] as [number, number[]];
    if (h >= a[0] && h <= b[0]) {
      const f = (Math.sqrt(h) - Math.sqrt(a[0])) / (Math.sqrt(b[0]) - Math.sqrt(a[0]));
      return (a[1][tauIdx] as number) * (1 - f) + (b[1][tauIdx] as number) * f;
    }
  }
  return last[1][tauIdx] as number;
}

/** The house baseline row set: persistence with the fixed random-walk spread. */
function baselineRows(x: number, horizon: number): number[][] {
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) rows.push(Z.map((z) => price6(x + z * SIGMA0 * Math.sqrt(h))));
  return rows;
}

/** The forecast of one market: horizon rows of the five quantiles. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  const prices = m.prices;
  if (prices.length === 0) throw new Error(`market ${m.id} has no prices`);
  const x = prices[prices.length - 1] as number;

  // Too little history to learn from: the house baseline, unchanged.
  if (prices.length < 30) return baselineRows(x, horizon);

  const recent = prices.slice(Math.max(0, prices.length - WINDOW));
  const anchors = anchorDeviations(recent, horizon);
  if (anchors.length === 0) return baselineRows(x, horizon);

  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const row = QUANTILES.map((_, j) => {
      const par = (Z[j] as number) * SIGMA0 * Math.sqrt(h);
      return x + EMP_W * empiricalDev(anchors, j, h) + (1 - EMP_W) * par;
    });
    row.sort((u, v) => u - v);
    rows.push(row.map(price6));
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
