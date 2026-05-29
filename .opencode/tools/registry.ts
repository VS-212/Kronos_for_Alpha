import { tool } from "@opencode-ai/plugin"

export const top = tool({
  description: "Show top N trading strategies by Sharpe ratio from the strategy registry. Use when user asks about best strategies, top performers, champions.",
  args: {
    n: tool.schema.number().default(10).describe("Number of top strategies to show"),
    minSh: tool.schema.number().optional().describe("Minimum Sharpe filter"),
    maxDd: tool.schema.number().optional().describe("Maximum drawdown filter"),
  },
  async execute(args) {
    let cmd = `python3 -m src.strategies.registry --top ${args.n}`
    if (args.minSh !== undefined) cmd += ` --min-sharpe ${args.minSh}`
    if (args.maxDd !== undefined) cmd += ` --max-dd ${args.maxDd}`
    return (await Bun.$`${cmd}`.text()).trim()
  },
})

export const lookup = tool({
  description: "Look up a trading strategy by name or partial name in the strategy registry. Use when user asks about a specific strategy, its metrics, or to find strategies by keyword.",
  args: {
    name: tool.schema.string().describe("Strategy name or partial name to search for"),
  },
  async execute(args) {
    return (await Bun.$`python3 -m src.strategies.registry --lookup ${args.name}`.text()).trim()
  },
})

export const stats = tool({
  description: "Show strategy registry statistics: total strategies, filters, champion metrics, date range. Use when user asks about registry health, coverage, or summary.",
  args: {},
  async execute() {
    return (await Bun.$`python3 -m src.strategies.registry --stats`.text()).trim()
  },
})

export const discover = tool({
  description: "Scan the verified/ strategies directory and report which champion strategies are available. Use when user asks what verified strategies exist.",
  args: {},
  async execute() {
    return (await Bun.$`python3 -m src.strategies.registry --discover`.text()).trim()
  },
})

export const backtest = tool({
  description: "Run a single SBER backtest with custom parameters. Use when user wants to test a strategy with different PL, TP/SL, commission or compare against registry champions.",
  args: {
    name: tool.schema.string().describe("Strategy name for output and optional registry registration"),
    strategy: tool.schema.string().default("wf").describe("Signal source: 'wf' for built-in walk-forward, or path to signals.npy"),
    pl: tool.schema.number().default(12).describe("Profit horizon in bars (default 12)"),
    lk: tool.schema.number().default(500).describe("Lookback window size (default 500)"),
    comm: tool.schema.number().default(0.0).describe("Commission per trade (default 0.0)"),
    tpSl: tool.schema.string().default("default").describe("TP/SL mode: 'default', 'no_tp', or 'no_sl'"),
    register: tool.schema.boolean().default(false).describe("Append result to registry.json"),
  },
  async execute(args) {
    let cmd = `python3 -m src.cli.backtest --strategy ${args.strategy} --name "${args.name}" --pl ${args.pl} --lk ${args.lk} --comm ${args.comm} --tp-sl ${args.tpSl} --json`
    if (args.register) cmd += " --register"
    return (await Bun.$`${cmd}`.text()).trim()
  },
})

export const compare = tool({
  description: "Compare two SBER backtest strategies side-by-side. Use when user wants to A/B test a new strategy against a reference champion or another signal.",
  args: {
    ref: tool.schema.string().describe("Reference: strategy name from registry or path to signals.npy"),
    test: tool.schema.string().describe("Test: strategy name from registry or path to signals.npy"),
    pl: tool.schema.number().default(12).describe("Profit horizon in bars"),
    tpSl: tool.schema.string().default("default").describe("TP/SL mode: 'default', 'no_tp', 'no_sl'"),
  },
  async execute(args) {
    return (await Bun.$`python3 -m src.cli.compare --ref "${args.ref}" --test "${args.test}" --pl ${args.pl} --tp-sl ${args.tpSl}`.text()).trim()
  },
})
